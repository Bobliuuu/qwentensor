# Exact layer / head / unit skipping on the chip path

Status: written before the implementation, numbers are filled in from `results/` as they are measured.

The idea comes from the qwendoom project (github.com/utsav1033/qwendoom, `model/stepper.c`): on any one token most of the
compiled model's layers do nothing, that can be decided from the weights and the current inputs, and leaving that work out
does not change a single emitted token.  This document is our own derivation for the qwentensor / qwenTPU chip layers; no
code is shared with qwendoom.

## 1. What the chip path computes per token

A chip layer is a stock `Qwen3_5DecoderLayer` (Gated DeltaNet token mixer + SwiGLU MLP) whose weights are emitted by
`qwentpu/compiler/emit.py`.  On one token (T = 1) the layer does, in this order and in these dtypes:

1. `x = rmsnorm(h)` (fp32 inside, result bf16; the gain is 1, the chip dims reach the projections scaled by 2^-k)
2. `qkv = in_proj_qkv(x)`, `z = in_proj_z(x)`, `b = in_proj_b(x)`, `a = in_proj_a(x)` (bf16 matmuls, fp32 accumulate)
3. causal conv (kernel 4, depthwise) over the cached last inputs and `qkv`, then `silu` -> `q, k, v` (bf16)
4. `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)` (fp32; `A_log = 0`, `dt_bias = -110`)
5. per head: `q, k` l2-normalised, `S <- exp(g) S`, `S <- S + k (beta (v - S^T k))^T`, `o = S^T q` (fp32)
6. `o` gated RMS norm with `silu(z)`, `out_proj`, residual add
7. `x2 = rmsnorm(h1)`, `act = silu(gate_proj x2) * up_proj x2`, `down_proj act`, residual add

The circuit is integer exact by construction (every value is a small integer held exactly in bf16, sums stay below 2^24).

## 2. Three kinds of dead work, with their proofs

**(S) Structurally dead (weights only, true for every token).**  A head whose `q` rows and `k` rows in `in_proj_qkv` are
all zero has `q = k = 0` after the conv and `silu` (`silu(0) = 0`, conv of zeros is 0), so `l2norm(0) = 0 * rsqrt(eps) = 0`:
it never writes its state (`S + 0 * delta`) and reads `o = S^T 0 = 0`.  Its contribution to the residual is exactly +-0.
An MLP unit whose `gate_proj` and `up_proj` rows are zero has `act = silu(0) * 0 = 0`.  Both are decided once, at load.

**(T) Dormant on this token (exact test, no bound).**  For the structurally live heads, compute the same `q`, `k`
pre-activations the dense layer computes (the same dot products over the same `x`, only fewer rows) and the same conv and
`silu`.  If every `q` and `k` channel of every head is exactly zero after `silu`, then by the argument above no head reads or
writes this token, and with `a`-driven decay equal to 1 (checked the same way: `exp(g) == 1` in fp32) the recurrent state is
unchanged.  If additionally every structurally live MLP unit has `act == 0`, the layer returns its input bit for bit.
The test costs a handful of small matmuls instead of the whole layer; it is exact because it is the dense computation on a
subset of rows, not an estimate.  (qwendoom proves the same fact one step earlier, from weight min / max bounds, so it
need not compute the rows at all; our rows are few enough that the exact test is cheaper than the bound machinery.)

**(V) Output vocabulary.**  The chip rows of `lm_head` read only the code dims, which are the only dims with a non-zero final
norm gain (all others have `1 + w = 0`).  Columns of the head that are all zero, or that meet a zero gain, contribute exact
zeros, so the readout is a matmul over the live columns only.

## 3. What must still happen on a dormant token

* The conv history: a channel with a tap at lag >= 1 reads inputs of earlier tokens, so its `qkv` input must be pushed into the
  cached conv state even when the layer does nothing else.  Channels with only a lag-0 tap never read history, so their cached
  values are irrelevant (their taps at lag >= 1 are exactly zero, and `0 * finite = 0`).
* The recurrent state is unchanged only if `exp(g) == 1` for every structurally live head (proved by the same exact test).

## 4. bf16 / fp32 exactness of the skipped computation

Dropping rows from a matmul does not change any other row's dot product, so the values the kept rows produce are the values the
dense layer produces **provided the kernel accumulates the same terms exactly**.  Our values are small integers (|partial sum|
< 2^24, every product exactly representable), so fp32 accumulation is exact in any order and bf16 rounding of an integer
is the identity; the final results are therefore independent of the kernel's tiling.  We do not rely on this argument alone:
section 6 compares emitted tokens, final hidden states and the cached conv / recurrent states of the skipping path with the dense
path on every real-model case.

## 5. Where the gain can come from, and where it cannot

The chip stack is a few kernels per layer on a small hidden state; on the A100 the eager path is bound by kernel launches and
the CUDA-graph path by the number of small kernels.  Skipping work helps in proportion to the *kernels* removed, not the bytes.
`results/` records what was measured.

## 6. Measurements

(filled in from `results/a100_skip_*.log`)

## Credit

The observation that a Qwen3.5-architecture model running a compiled machine leaves almost all layers idle on each token, and
that this can be decided exactly, is qwendoom's (`utsav1033/qwendoom`).  This implementation is independent.
