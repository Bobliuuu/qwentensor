# qwentensor integer graph IR (contract between frontend/quantizer and the machine executor)

Code: `qwentensor/ir.py` (container, validation, cost model, JSON), `qwentensor/reference.py` (bit-exact numpy golden).
The golden is `reference.run(graph, encoded_input)`. The executor is correct iff it reproduces that output exactly.

## 1. Machine facts the IR is built on (from qwenTPU v2, `docs/V2_ISA.md`)

* Array `KR x NC` (default 16x16), activations unsigned `ab` bits = `0..31`, weights signed `wb` bits = `-16..15`, exact int accumulate.
* One machine matvec = `a = x @ tile` (+ bias as a weight on a constant-1 row, so a tile has `KR-1` input rows and `NC` columns;
  a non-final tile also loses a column to the constant-1 generator when the result is chained in the model's own tokens)
  followed by the fused nonlinearity `clip(relu(a) >> shift, 0, 31)`. **v2.0 hardware: shift = 0 only. v2.1 (simulator-only gadget
  `qwentpu/compiler/nonlin.py`): shift 0..15.**
* Everything else (layout, token order, weights, input) is prompt data written by the host.

## 2. Graph

`Graph(nodes, output, argmax, profile, ab, wb, bias_rows)`; nodes are topologically ordered. **Every tensor is a flat 1-D integer vector**
(batch is outside the graph; the machine runs one sample per program). Exactly one `input` node.

| op | fields | semantics | machine cost |
|---|---|---|---|
| `input` | `out_len`, `encode`, `vocab`, `scale` | host builds the vector from the raw sample (see below); values in `0..amax` | prompt tokens |
| `gather` | `inputs=[x]`, `index[out_len]` | `out[i] = x[index[i]]`, `index[i] == -1` -> 0. Covers im2col, reshape/flatten (these never emit a gather: the frontend re-indexes the consumer's weights), permute, duplicate, zero-pad | free for the host if its input is the program input; on an intermediate it is a re-layout (see 5) |
| `concat` | `inputs=[a,b,...]` | `out = [a; b; ...]` | free (it is the row order of the next tile) |
| `matvec` | `inputs=[x]`, `W (K,N) int`, `b (N) int`, `shift`, `act`, `groups P` | below | `ceil(K/(KR-bias_rows)) * ceil(N/NC) * P` tile matvecs |

`matvec` semantics, exact int64 (a shared `W` is applied to each of the `P` consecutive `K`-slices of the input):

```
acc[p, n] = sum_k x[p*K + k] * W[k, n] + b[n]            # no overflow, exact
act = relu_clamp (machine native):  out[p*N + n] = clip(acc[p, n] >> shift, 0, 31)     # >> is arithmetic = floor
act = linear     (NOT machine-native): out = acc >> shift                              # signed, host-read only
```

Nothing else exists. In particular there is **no ADD op**: a residual `relu(f(h) + x)` is `concat[f_pre-acts' inputs..., x]` feeding ONE matvec whose weight
rows for `x` are a (scaled) identity, so the add is accumulated in the array for free and the nonlinearity is the same fused relu_clamp.
There is **no separate pool op**: AvgPool is a fixed matvec with weights `1/k`, composed into the NEXT matvec when one follows (exact in float,
quantized once), or emitted as a grouped matvec on its own (`W = 1/k`, power-of-two k uses `shift`) when a conv follows.
`Embedding` on the integer input is a matvec on a one-hot `input` (`encode="onehot"`).

**Layout rule.** A grouped matvec writes its output position-major: `out[p*N + n]`. `Flatten`/`reshape` never move data; the consumer's
weight rows are indexed in that physical order (the frontend re-indexes them), so a Linear after a Conv sees rows `p*Co + co`, not torch's `co*P + p`.

### Ranges and constraints (checked by `Graph.validate(machine=True)`)

* `W` in `[wlo, whi] = [-16, 15]`; `b` in `bias_rows * [-16, 15]` (the executor spends `bias_rows` constant-1 rows per tile; default 1 so the usual
  v2 bias convention holds. Note the bias is in ACCUMULATOR units, i.e. before `shift`, so with shift s it is worth only 2^-s of an activation step); `shift` in `0..15`, and `0` in profile `v2.0`; activations `0..31`.
* Profiles: `v2.0` (shift == 0 everywhere, act == relu_clamp), `v2.1` (shift allowed), `float` (un-quantized graph, `act` in `relu|none`).
  `act == "linear"` is only produced when the user asks for signed logits; it is **not** executable by the v2 datapath (final relu clamps logits to
  `0..31`; negative logits become 0 and ties are possible). `Graph.validate(machine=True)` rejects it.
* `argmax=True`: the HOST takes `np.argmax` (lowest index wins ties) of the output vector. It is not a machine op.

### Input encoding (host, `reference.encode_input`)

* `uint`: `q = clip(round(x / scale), 0, 31)` (inputs must be >= 0; for signed inputs the frontend uses `posneg`).
* `posneg`: `[clip(round(relu(x)/s)); clip(round(relu(-x)/s))]`, first-layer weights are `[W; -W]` (doubles the input width).
* `onehot`: integer token ids `(T,)` -> `T*vocab` 0/1 vector (an `nn.Embedding` becomes a matvec on it).

## 3. How each PyTorch op lowers

| PyTorch | status | lowering |
|---|---|---|
| `nn.Linear`, `F.linear` | supported | dense term of a matvec; `Linear->Linear` with no ReLU between is merged into one matvec (exact in float) |
| `nn.ReLU`, `F.relu`, `.relu()` | supported | the fused `relu_clamp` of the matvec that produces its input (ReLU of an already-activated tensor is a no-op) |
| `nn.Conv2d` (groups=1, dilation=1, zero padding, any stride) | supported | `gather` (im2col, zero padding = index -1) + `matvec` with `groups = Ho*Wo`, `W = (C*kh*kw, Co)` shared across positions |
| residual `torch.add` / `+` | supported | extra identity rows in the consuming matvec over `concat[...]`; works for flat features (not across Conv outputs) |
| `nn.AvgPool2d`, `AdaptiveAvgPool2d`, `F.avg_pool2d` | supported | fixed matvec, folded into the next Linear (K grows to the pre-pool size) or a grouped matvec |
| `nn.Flatten`, `.view`, `.reshape`, `torch.flatten` | supported | free: logical shape change only, the consumer's weight rows are re-indexed (no gather) |
| `nn.Embedding` on the network's integer input | supported | one-hot matvec (`K = T*vocab`); rows of the table can be negative only if a Linear follows directly (ReLU of a signed embedding would be required otherwise) |
| `nn.Identity`, `nn.Dropout` (eval) | supported | nothing |
| `torch.argmax` / `.argmax()` as the last op | supported | `argmax=True` (host) |
| `nn.Softmax`, `log_softmax` | **NOT** | trailing softmax before argmax: drop it. Otherwise needs exp: a LUT gadget (piecewise-linear over the 32 input levels, e.g. `exp` in 32 steps as a ReLU basis `sum_j c_j relu(x - j)`: one matvec + one relu_clamp layer, ~1 more layer) + a reciprocal of the sum (second LUT) + a data x data product (no) -> approximate with the max-subtracted `exp` LUT and keep the argmax in the host |
| `nn.LayerNorm`/`BatchNorm` | **NOT** | BatchNorm in eval folds into the preceding Linear/Conv (supported by folding, not done automatically yet). LayerNorm needs mean/var/rsqrt/division per sample: mean is a matvec; `1/sqrt(var)` is a LUT on a quantized variance and the multiply is data x data (not available). Possible later as RMS-free / fixed-scale norm folded into weights |
| `nn.GELU`, `SiLU`, `Sigmoid`, `Tanh` | **NOT** | piecewise-linear approximation with k ReLU units (`sum_j c_j relu(x - t_j)`): one matvec to compute `x - t_j` and a second to sum, 2 layers per activation; exactness limited by 5-bit activations (32 levels), so a 32-entry LUT is exact for any scalar function of a quantized value |
| attention (`MultiheadAttention`, `matmul` of two activations) | **NOT** | needs softmax and a data x data product (weights are stationary in the array); a 32-level LUT for products `x*y` (1024 entries, or 2 ReLU-squares: `xy = ((x+y)^2-(x-y)^2)/4`) would cover it in principle at large cost |
| `nn.MaxPool2d` | **NOT** | `max(a,b) = a + relu(b-a)` is expressible (2 layers); not implemented. Use AvgPool or stride-2 conv |
| elementwise `mul` of two activations | **NOT** | same as above (data x data) |
| negative activations in general | limited | activations are unsigned; only the input has the `posneg` split, interior tensors are post-ReLU so non-negative |

Unsupported ops raise `qwentensor.frontend.UnsupportedOpError` naming the op, the fx node and a hint.

## 4. Quantization contract (`quant.py`)

Real value of a tensor element = `scale[i] * q[i]`. For a matvec with float `Wf`, `bf`, input scales `s_k`, output scale `s_o`, shift `sh`:
`Wq[k,n] = clip(round(Wf[k,n] * s_k * 2^sh / s_o))`, `bq[n] = round(bf[n] * 2^sh / s_o) (+ 2^(sh-1) for round-to-nearest)`. So the per-layer
scale is folded into the integer weights **and** the shift; `s_o` and `sh` are chosen per layer by a calibration search that minimizes the
layer's output error on sample data. Profile v2.0 forces `sh = 0`, which forces `s_o` to be tied to the weight range (the documented v2.0 limitation).

## 5. What the executor must provide (and what is open)

1. Tiling: a matvec with `K > KR-1-bias_rows` or `N > NC` is split into tiles; partial sums over K must be summed exactly BEFORE `shift`/relu/clamp
   (the fused op cannot be applied per tile). That needs either an un-clamped signed partial-sum readout or the bias-row trick; it is the executor's design.
   `Graph.cost()` gives the tile counts.
2. `groups = P`: the same weight tile is reused for P input slices.
3. `concat`: rows of the next tile, free. `gather` on the program input: free (prompt layout). `gather` on an intermediate (conv after conv):
   either re-prompt from host between stages (host reads OUT tokens, builds the next prompt) or duplicate/permute weight rows of the consumer.
4. Bias magnitude up to `bias_rows*16` needs that many constant-1 rows (`bias_rows` is in the graph).
5. Output: `argmax` is host-side.
