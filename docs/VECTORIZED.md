# Vectorized execution: several neurons per emitted token

Code: `qwentensor/machine/vec.py` (spec, circuit, program, scheduler, CPU machine). The scalar machine
(`spec.py`, `circuit.py`, `asm.py`, `sched.py`) is unchanged and stays the default.
Tests: `tests/test_machine_vec.py`. GPU: `scripts/gpu_run.py vec`, `vecbench`.

## 1. Why the scalar machine is slow

One neuron costs two emitted tokens: `READ(j)` and `ACT(dest; v)`.
The chip runs one full forward pass (9 to 16 layers) per token, so decode time is `tokens x layers`.
The digits MLP (25 neurons) emits 50 tokens.

The two tokens do different jobs:

* `READ(j)` is the token at which the accumulator heads are queried. Its position computes
  `requant(sum of the masked columns)` and its output spells `ACT(dest; v)`.
* `ACT(dest; v)` is the token that writes the value into register `dest` (accumulator rows, via the weight-product FFN).
  Its output is only the successor `READ(next)`.

## 2. What the circuit can do in one token

Three facts from the circuit IR (`qwentpu/compiler/evaluate.py`, `eval_head`) decide the design.

1. **A query sees the write of its own position.** `eval_head` applies the key write at position `t` and then reads the state
   for the query at `t`. On Qwen3.5's Gated DeltaNet this is the delta rule `S_t = S_{t-1}(I - beta k k^T) + beta v k^T`, `o_t = S_t q_t`.
   So the token that *writes* a register can also *query* the accumulators with the new value already inside.
2. **Token features are free functions of the token id.** Anything the token id determines (a row number, a tile, the bits of
   several values) is a feature dim at no cost: the embedding row holds it.
3. **A head has one key per token, but different heads have different keys.** A column's accumulator head is written only if
   the token's `beta` gate for that column's tile is on. Two values written by one token therefore need two *different tiles*, not two heads.

### 2.1 Fusion (L = 1): READ and ACT become one token

Program memory is keyed by the *current* token's register instead of by an op slot. The token `ACT(r; v)` writes `v` to
register `r` and, in the same position, queries the accumulators with the mask of the op stored under slot `r`.
Its output is `ACT(dest; v')`, the next register write. There is no `READ` token.

Predicted token reduction: 2 per neuron to 1 per neuron, so 2x fewer sequential steps. The circuit is the same size, so tokens/s is unchanged and
inference time halves.

### 2.2 Lanes (L > 1): one token carries L values

A *group* is L registers that sit at the same row of L different tiles (lane l of block b is tile `b*L + l`).
The token `VEC(g; v_0 .. v_{L-1})` writes lane l's value into row `g mod rows` of tile `(g div rows)*L + l` for every l:
each column's accumulator head has exactly one write (its tile's lane value), so nothing is duplicated.

* **Columns are partitioned by lane class.** A column belongs to the neuron that owns it, and a neuron emits its value in lane `l`.
  The adder tree and requantizer are instantiated once per lane (FFN units grow with L, depth does not).
  A token id is `VEC0 + g*nout^L + sum_l v_l nout^l`; the value bits are the low `L*ab` bits and the base is aligned, so
  the next-token code is `succ_base + sum_l (requant bits of lane l)`, carry free.
* **Independent neurons per op group.** The L neurons of one op group must not depend on each other. The scheduler pairs
  neurons of the same layer.
* **Cost.** A neuron that reads a layer whose registers are spread over the L tiles of a block needs one column per touched
  tile. A 15-wide layer occupies 8 rows of two tiles instead of 15 rows of one, so such a neuron has 2 columns, not 1.
  Heads are 2 per column, so heads (and layers) grow with L.
* **Vocabulary.** `VEC` tokens: `ngroups * nout^L`. With u5 and L = 2 this is 1024 per group (32768 for 32 groups);
  L = 3 would be 32768 per group and is not built.

### 2.3 Prompt-side writes

A `VEC` token queries. The accumulator heads are normalised by the number of rows written; that is exact only for a full tile
(16 rows). So the prompt initialises every register with a single-register `ACT(r; v)` token (writes, never queries), and
the last prompt token is one `VEC` token for the entry group, after every other row has been written.
This keeps every query in the run on a complete tile and keeps all evaluator exactness conditions true at every position.

## 3. Exactness

Unchanged mechanism: counts of the selected columns are added in one adder tree before shift, relu and clamp, in bf16-exact
integers (`fan * rows <= 256`). Per lane the same window limits apply. The independent numpy token golden
(`vec.reference_emit`) parses the prompt, applies each token's writes, and then computes the successor group.
The CPU evaluator asserts every bf16 / silu / one-hot / equal-multi-hot condition at every position, including the new fused query positions.

## 4. Predicted effect (digits MLP 15-15-10, 25 neurons) and what was measured

| machine | emitted tokens | circuit |
|---|---|---|
| scalar | 50 | 9 chip layers, 25 columns |
| fused, L = 1 | 25 | same size |
| lanes, L = 2 | 13 | about 2x the columns, so more layers |

Measured on the real Qwen3.5-0.8B (A100, bf16, digits MLP, 20 inputs, all exact; logs `results/a100_vec_{exact,general,bench}.log`):

| machine | chip layers | hidden | vocab | emitted | eager loop / end to end | CUDA-graph loop / end to end |
|---|---|---|---|---|---|---|
| scalar | 9 | 3072 | 17616 | 50 | 656 / 680 ms | 163 / 266 ms |
| 1 lane | 9 | 2816 | 19744 | 25 | 331 / 355 ms | 101 / 205 ms |
| 2 lanes | 17 | 5120 | 71840 | 13 | 322 / 367 ms | 146 / 340 ms |

* **Exactness:** 20/20 digits x 2 readouts for 1 and 2 lanes; a 16-layer one-lane machine ran four programs 24/24.
* **One lane** is a pure win: half the tokens, no extra layers. 2.0x eager, 1.6x in the graph loop, 1.3x per inference (prefill dominates the graph path).
* **Two lanes** halve the tokens again, but the columns double (a neuron reads registers in both lane tiles, so it needs one column per tile), heads go from 53 to 121 and layers from 9 to 17. The cost per token roughly doubles, so the gain is lost:
  a tie eagerly and slower with CUDA graphs. `block_align=True` (each node on a fresh block) gives 16 layers and 103 heads for two lanes; it was not run on the GPU.
* **Not tried:** three or more lanes (vocabulary `nout^L` per group is 32768 ids per group at three lanes).

## 5. What is not changed

No loops or data-dependent branches. The group chain is still fixed by the prompt. A neuron wider than `fan * (rows - 1)` inputs still does
not fit. The scalar machine, the pages (`machine/pages.py`) and every existing test are untouched.
