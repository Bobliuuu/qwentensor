# qwentensor design

This is the readable version of the design. The precise contracts are in [IR.md](IR.md) (the integer graph) and [EXECUTOR.md](EXECUTOR.md)
(the machine inside Qwen3.5). Where this note and those disagree, those win.

**Status in one paragraph.** The frontend, quantizer and the numpy golden (`reference.run`) are done and tested. The executor is verified on a CPU
simulator of the circuit (qwenTPU's float64 circuit-IR evaluator, with every bf16 / silu exactness condition asserted): its output is bit-exact
against `reference.run`. On the real model, one network has run: the quantized digits MLP 15-15-10 on Qwen3.5-0.8B (A100, bf16), 20 held-out digits x 2 readouts,
40/40 exact against the numpy token golden and `reference.run` (`results/a100_digits.log`). Then one 14-layer machine (rows=16, cols=(15,12,8), fan 2, smax 7, 40 ops, vocab 26616) ran four different programs (digits MLP, a K=20 two-tile MLP, a 4-layer MLP, a residual block), 24/24 exact, only the prompt differing (`results/a100_general.log`). The deep MLP, the CNN and the frontend's residual MLP have not run on the real model.

## 1. The problem

qwenTPU compiled a weight-stationary integer matrix-multiply array into the layers of a stock Qwen3.5. The three versions differ in who decides
what the machine computes:

| | what runs | who decides the computation |
|---|---|---|
| qwenTPU v1 | one fixed matvec (`x @ W`, 5-bit unsigned by 5-bit signed, exact) | the host, one operation at a time |
| qwenTPU v2 | a hardcoded chain of matvec + requantize stages; the prompt varies widths and depth | the compiled-in chain; the program only chooses how long it is |
| qwentensor | any network the supported-op list covers, written as a PyTorch `nn.Module` | the prompt: successor, destination register, column mask and shift of every op are tokens |

The question qwentensor answers is: can the thing you give the machine be an ordinary PyTorch network, with the schedule living in the prompt, and
the host doing nothing but `argmax; append`? For small networks, within the limits in section 9, yes, on the simulator; on the real model it has been shown for the digits MLP only.

## 2. The pipeline

```
nn.Module --torch.fx--> float IR --quantize (PTQ, optional STE QAT)--> integer IR --+--> reference.run (numpy, bit-exact golden)
                                                                                    +--> sched.compile_graph --> Spec + Program --> prompt --> Qwen3.5 emits tokens --> decode
```

`qt.compile(model, calib, input_shape=...)` runs the whole chain and returns a `Program`. `Program.run(x)` encodes the input on the host, builds the
prompt, runs the machine (backend `sim` today, `qwen` on a GPU) and decodes the emitted tokens; by default it asserts the decoded output equals
`reference.run`.

## 3. The IR

The IR (IR.md section 2) is deliberately tiny: every tensor is a flat 1-D integer vector, and there are four ops.

* `input`: values `0..31`, built by the host from the raw sample (`uint`, `posneg` for signed inputs, `onehot` for token ids).
* `gather`: `out[i] = x[index[i]]`, index -1 gives 0. This is im2col, permute, zero-pad.
* `concat`: free; it is the row order of the next tile.
* `matvec`: `acc = x @ W + b` in exact integers, then `out = clip(acc >> shift, 0, 31)`. A shared `W` can be applied to `P` consecutive slices
  (`groups`), which is how a convolution is expressed.

There is no add, no pool, no flatten. A residual is extra identity rows of the consuming matvec over a `concat`, so the add happens in the
accumulator for free. Average pooling is a fixed matvec that is folded into the next layer's weights in float and quantized once. Flatten and
reshape never move data; the consumer's weight rows are re-indexed. A trailing `argmax` is a host operation.

Why so small: the machine can do exactly one thing per step (a matvec with a fused nonlinearity), so the IR is that thing and everything else is
rewritten into it by the frontend. `Graph.validate(machine=True)` checks the ranges the hardware has.

## 4. The quantizer

`quant.py` turns the float IR into the integer IR. A tensor element's real value is `scale * q` with `q` in `0..31` (activations, unsigned) and
weights in `-16..15`. For a layer with float weights `Wf`, bias `bf`, input scales `s_k`, output scale `s_o` and shift `sh`:

```
Wq = clip(round(Wf * s_k * 2^sh / s_o)),   bq = round(bf * 2^sh / s_o)   (+ 2^(sh-1) to round to nearest)
```

so the output scale is folded into the integer weights and the shift. `s_o` and `sh` are picked per layer by a calibration search that minimizes
that layer's output error on the calibration samples. Post-training quantization (PTQ) is the default; `quant.qat` fine-tunes the integer
weights with a straight-through estimator, model selection on train accuracy only. Scales are per tensor.

Two hardware profiles matter, and the accuracy depends on them heavily:

* `v2.0`: the shift is 0 everywhere, as in the shipped qwenTPU v2 datapath. Then `s_o` is tied to the weight range, and PTQ collapses on anything
  but a tiny network (24-49% on the deeper digits MLPs, 14% on the CNN; section 8 of the README has the table).
* `v2.1`: a per-op shift of 0..15. This is the requantize gadget `qwentpu/compiler/nonlin.py` plus a program-memory field, and it is what
  `qwentensor.machine` implements (shift 0..`smax` chosen per op by the prompt). It is the main accuracy lever.

## 5. The executor: a register machine with the program in the prompt

Full detail in EXECUTOR.md. The short version:

* **Registers.** `tiles` banks of `rows` registers, each holding `0..31`. The top `bias_rows` rows of each tile are constant-1 registers.
* **Neurons are columns.** A column holds one signed weight per row of its tile, as resident state loaded by `CELL(row, col, w)` prompt tokens
  (zero weights need no token). A neuron is one op.
* **One op = one `READ`.** `READ(j)` sums the accumulator counts of the columns op `j` names, applies `clip(floor(y / 2^shift_j), 0, 31)`, and
  emits `ACT(dest_j; value)`. The emitted value token is the next register write.
* **Program memory.** Per slot, the words `succ` (which token comes next), `mask` (which columns), `shift` live in DeltaNet head state, written by
  `PROG` tokens in the prompt. The current token addresses its slot, so the next token is `RAM[current token]`. No extra layers are spent on this:
  the same multi-hot READ machinery that reads weights reads the program.
* **Host loop.** `argmax; append` until `HALT`. The host also encodes the input and decodes the output vector from the emitted `ACT` tokens;
  there is no host arithmetic on the data in between.

A prompt is `CELL* PROG* ACT(init registers)* ACT(entry)`.

What is general: which op feeds which (any DAG, any register-reuse pattern the single-assignment rule allows), the op count up to `nops`, each op's
fan-in columns, shifts, weights, widths, depth. What is compiled in: the machine geometry (`rows`, columns per tile, `fan`, `smax`, `nops`, bit
widths), the token layout, and the fact that each op is one READ followed by one ACT with relu+clamp as the only nonlinearity. The test
`test_machine_prog.py` compiles one circuit and runs 12 different programs on it (different slot numberings and orders, with and without skip
connections, depths 1 and 5).

## 6. Tiling and partial sums

A layer wider than one tile of `rows - 1 - bias_rows` inputs and `NC` columns is split. Output tiling is free: each neuron is its own op. Input
(K) tiling is where a fused `relu(clamp(...))` would be wrong, since partial sums must be added before the nonlinearity.

Instead of passing partial sums as digit tokens, qwentensor uses a property of the accumulator head: it returns exact bit-plane *counts*, and counts
of different columns just add. A neuron whose inputs sit in several tiles is `fan` columns, one per tile; the op's mask lights all of them, and
one adder tree adds their counts before any shift, relu or clamp. Partial sums are never clamped and never leave the datapath. A tile's rows are
written only by that tile's `ACT` tokens, so the same 16 key rows serve every tile.

Cost: wider digits in the tree, so more chip layers (rows=16, fan 1: 7 layers; fan 2: 10; with shifts up to 4: 13). Limits: counts must stay
bf16-exact (`fan * rows <= 256`) and the carry staircase window of the requant gadget must hold the digit ranges (`fan * rows` up to about 48 at
5-bit; rows=16, fan=3 builds, rows=32, fan=2 does not). A neuron that needs more than `fan * (rows - 1)` inputs, for example a 784-wide first layer,
is **not solved**; see section 9.

Convolutions: `groups = P` duplicates the weight columns per output position, so a conv layer costs `P` times the neurons.

## 7. Requantization by shift

The array produces `y`; the layer wants `clip(floor(y / 2^s), 0, 31)` with `s` chosen per op. The compiled-in gadget `nonlin.requant` has a fixed
shift `smax`. To get a variable one, the program-memory shift field (one-hot, read at level 1) selects which digit positions of the counts are used,
forming `y * 2^(smax - s)`, and the fixed `requant(shift = smax)` then yields `floor(y / 2^s)` exactly. Digits are per bit-position `a + b`
(weight `2^p`, sign from the top weight bit), which keeps each digit's range near `5 * fan * rows` inside the carry window. Checked against numpy on
random programs and on worst cases (all weights at +15 / -16, all activations 31, shifts 0..7) at 3-bit and 5-bit.

## 8. Bias rows

The bias is one weight on a constant-1 row, so with 5-bit weights it lies in `[-16, 15]` in *accumulator* units. After a shift `s` that is worth
only `2^-s` of an activation step, which clipped 40-60% of the biases of the small digits network. `bias_rows = b` spends `b` constant-1 rows per
tile and widens the range to `b * [-16, 15]`; the bias is split over those rows in the neuron's first column. The trade is `b` fewer input rows per
tile. On the 15-15-10 digits net, 4 bias rows move v2.1 PTQ from 76.3% to 85.2% (README, results).

## 9. What is general, what is limited

General: any feed-forward DAG of the supported ops, any widths and depth, per-layer shifts, residual connections, convs and pooling through the
rewrites above, one-hot embeddings of integer inputs, all expressed in the prompt.

Limits (honest list):

* **Size.** Heads grow about 2 per column, 16 key heads per layer; about 100 neurons is the practical ceiling on the 0.8B model. At most 128 ops
  and 126 physical columns (`qt.DoesNotFit` explains which). The 64-96-64-32-10 deep MLP in `examples/` has 202 neurons, so it is a quantizer
  result and an IR-level example, not a machine one.
* **No loops, no data-dependent branches, no early exit.** Registers are single-assignment (a register's successor is a property of the register),
  so program length is bounded by the register count. Keying the successor on (register, value) would allow branches; not built.
* **Wide first layers.** A neuron with more than `fan * (rows - 1)` inputs (784 pixels, for example) needs partial sums across time (an activation
  RAM, or host-sequenced READs). Not built. The circuit builds `fan * rows` up to 48 when shifts reach 5 or more and up to 80 when `smax <= 4`
  (`scripts/probe_fan.py`), so with trained shifts a neuron reads at most about 45 registers; the 64-input examples do not fit even in pages.
* **Weight streaming is host-paged, not in-model.** Inside one run all weights are resident in state. A network with more neurons than one machine runs as
  pages (`qt.compile(..., paged=True)`, `docs/STREAMING.md`): the model runs each page from its own tokens and the host copies the integer activations
  between pages.
* **Signed interior activations.** Activations are unsigned and post-relu. Only the input has the `posneg` split. A final `relu` clamps logits
  to `0..31`, so ties are possible; `act="linear"` (signed logits) is not machine-native.
* **Accuracy.** 5-bit weights and activations, per-tensor scales. See the README table for the loss per network.
* **Speed.** One generated token per neuron; this is a demonstration of in-weights computation, not an accelerator.
* **Real model.** Only small programs have run there: the 9-layer digits machine and a 14-layer, 40-op general machine. Anything near the ~100 neuron ceiling has not.

## 10. What is not covered, and how it could be

Each item is a design sketch, not code. The common tool is that activations have only 32 levels, so any scalar function of a quantized value has a
32-entry table, and a table is a ReLU basis: `f(x) = sum_j c_j relu(x - t_j)`, one matvec to form `x - t_j` and one to sum.

| op | why it is missing | how it could be added |
|---|---|---|
| GELU, SiLU, sigmoid, tanh | not relu | a 32-entry table as a ReLU basis: exact for any scalar function of a 5-bit value, about 2 extra layers per activation |
| MaxPool | not linear | `max(a, b) = a + relu(b - a)`, two layers; or use AvgPool or a stride-2 conv today |
| BatchNorm (eval) | not done automatically | folds into the preceding Linear/Conv in float; needs an fx pass |
| softmax | exp and division | a trailing softmax before argmax is dropped. Otherwise a max-subtracted `exp` table plus a reciprocal table plus a data x data product |
| LayerNorm | mean, variance, rsqrt, multiply | mean is a matvec, `1/sqrt(var)` a table on a quantized variance, but the final multiply is data x data; a fixed-scale norm folded into the weights is the realistic variant |
| data x data products (attention, elementwise mul) | weights are stationary | `xy = ((x + y)^2 - (x - y)^2) / 4` with a square table (a ReLU basis), or a 1024-entry table; large cost |
| attention | needs softmax and products | the two rows above together; only worthwhile at very small sizes |
| loops, recurrent layers | no control flow | unrolling in the schedule is possible within the register budget; real loops need a counter and a successor keyed on (register, value) |
| wide first layers | K beyond `fan * (rows - 1)` | partial sums across time: activation RAM that re-emits `ACT` tokens, or prompt-interleaved READs |

## 11. Verification status

* 46 CPU tests (`OMP_NUM_THREADS=1 pytest -q tests`, about 1 min; all pass. `test_quantized_close_to_float` checks the quantizer, not the machine: on seeded random inputs and random-weight nets the residual MLP correlates about 0.97 with float (bar 0.95) and the CNN about 0.89 (bar 0.85); the trained nets in `examples/` are scored by accuracy instead). The tests include: program memory, tiling, ops at 3-bit and 5-bit extremes, scheduler (concat, gather, groups,
  `bias_rows = 2`), and the real quantized digits MLP 15-15-10 with shifts 2 and 4, all equal to `reference.run` in the circuit-IR simulator.
* Real Qwen3.5-0.8B, A100, bf16: the digits MLP, 40/40 exact (`results/a100_digits.log`); and one 14-layer machine running four programs, 24/24 exact (`results/a100_general.log`). Not verified there: the deep MLP, the CNN, the frontend's residual MLP, other Qwen3.5 sizes.
