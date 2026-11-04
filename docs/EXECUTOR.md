# qwentensor executor: a register machine with a prompt-written program, inside Qwen3.5

Code: `qwentensor/machine/` (`spec.py` token layout, `circuit.py` the circuit in qwentpu IR, `asm.py` program/prompt + independent
token-level numpy reference, `sched.py` Graph -> program, `run.py` host loop + CPU simulator driver, `gpu.py` hook into qwentpu merge).
Tests: `tests/test_machine_*.py` (CPU, qwentpu's float64 circuit-IR evaluator with every bf16 / silu / state-row exactness
condition asserted). GPU: `scripts/gpu_run.py`. qwenTPU itself is untouched (imported; `gpu.py` only rebinds `qwentpu.circuit.build/VOCAB` at runtime).

## 0. Summary (honest)

| problem | status |
|---|---|
| 1 program memory: schedule is prompt data | **done, exact** (sim + A100). Successor, destination register, column mask and shift of every op are written by `PROG` tokens and read back by the existing multi-hot READ machinery; next token = RAM[current token]. Zero extra layers. Arbitrary DAG order / slot numbering / skip connections / depth. No loops, no data-dependent branches. |
| 2 K tiling (wide partial sums) | **done, exact, but NOT by digit tokens**: the accumulator counts of up to `fan` columns (any tiles) are added inside one adder tree before any shift/relu/clamp. Partial sums are never clamped, never leave the datapath. Limit: `fan*rows <= 256`, window limits of the requant gadget (section 3). K beyond fan*(rows-1) per neuron is NOT supported. |
| 3 ops | relu_clamp (native), requant by shift 0..smax chosen **per op by the program memory** (exact floor, `nonlin.requant` fixed at smax + one-hot digit mux), residual add (as IR.md: identity rows, summed in the tree), bias_rows>1 (several constant-1 rows), concat/gather (host re-indexing; duplicates split over columns). |
| 4 scheduler + generator | `sched.compile_graph(Graph) -> Spec + Program`; `Program.prompt()` (tokens), `Program.expected()` (token-level golden), `Compiled.decode(emitted)`; host loop `run.generate` = argmax; append until HALT. |

Bit-exactness: every stage is checked against numpy in the CPU simulator, and the real quantized digits MLP (`reference.run`) on the real
Qwen3.5-0.8B on the A100 (section 7).

## 1. Machine model

* **Register file**: `tiles` banks of `rows` registers; register `r = tile*rows + s`; value `0..amax` (unsigned 5 bit by default). The top
  `bias_rows` rows of every tile are constant-1 registers (bias).
* **Columns** (neurons): `cols[t]` physical columns in tile t. A column = one weight per row of its tile (weight head keyed by row) plus an
  accumulator head keyed by row. Weights are resident, loaded by `CELL(s,c,w)` prompt tokens (zero weights need no token).
* **Op** = READ slot `j`. `READ(j)`: sums the accumulator counts of the columns named by op j's mask (`fan` mask fields), applies
  `clip(floor(y / 2^shift_j), 0, amax)`, and emits `ACT(dest_j; value)`: the value token **is** the next register write.
* **Program memory** (DeltaNet heads, level 1, no extra layer): per slot (op j, or register r) the stored words are
  `succ` (id of the successor token without the data part), `mask_f` (one-hot column, f < fan), `shift` (one-hot). `PROG` tokens write them
  in the prompt. The current token addresses them: `READ(j)` -> slot j, `ACT(r;v)` -> slot nops+r.
* **Next token** (code dims): `succ[slot(cur)] + (value bits if cur is READ)`. Alignment `ACT0 % NOUT == 0` makes the add carry-free.
  Unwritten slot => HALT (id 0).
* **Control flow** is therefore: op j's result goes to register `dest_j`; register r's successor is op `next(r)`; the prompt's last token is an
  ACT of the entry register. The host only supplies the prompt and runs `argmax; append`.

Prompt = `CELL*  PROG*  ACT(init registers)*  ACT(entry)`. No READ / emitted-class token, no host arithmetic.

## 2. Problem 1: program memory

Design B of `docs/V2_SEQUENCER.md` section 5, built. Differences from the sketch:
* a PROG token writes one *word* of one slot (`PS_READ(j,r)`, `PS_ACT(r,j)`, `PM(f,j,c)`, `PSH(j,s)`), so each head's key is the slot (one-hot, <=128
  per head; succ uses ceil((nops+nreg)/128) heads; mask and shift heads need nops <= 128). The delta rule replaces a whole row, so every field
  that must be set together lives in its own head.
* the READ's value bits are added with `ACT0 % NOUT == 0` (no carry feature).
* mask / shift are read at level 1 and consumed by the accumulator-head queries and the digit mux (which need level >= 3 anyway): **no layer cost**.

Exactness: single-row reads return the stored bits exactly (same pattern as the weight heads); stored bits = code bits (<= 15..16).
Tests: `test_machine_prog.py`: ONE compiled circuit runs 12 different programs (3 slot numberings and execution orders, with/without skip
connection, depths 1 and 5), emitted ids == numpy golden, everything the evaluator asserts (bf16, silu windows, one-hot keys, equal multi-hot
queries) holds.

**What is truly general**: which op feeds which (any DAG, any register reuse pattern allowed by SSA), op count up to `nops`, each op's fan-in
columns, shifts, weights, widths, depth. **Still compiled in**: the machine geometry (`rows`, `cols` per tile, `fan`, `smax`, `nops`,
`ab`, `wb`), the token layout, "one op = one READ that emits one ACT", relu+clamp as the only nonlinearity.
**Unsolved/limits**: (a) SSA: one op per destination register, because a register's successor is a property of the register (a loop that reuses a
register would need a counter); so program length <= number of registers; no loops, no early exit, no data-dependent branch (the successor
could be keyed on (register, value), 32x the keys, not built). (b) 128 slots per mask/shift head (more heads would be needed beyond 128 ops).
(c) the program must fit entirely in state: weights are resident, there is no weight streaming.

## 3. Problem 2: tiling and partial sums

Output tiling: more columns / more ops (each neuron is an op, so there is no output-width limit except heads).

K tiling. The task suggested digit tokens or un-clamped state. Neither is needed here: an accumulator head already returns *exact
bit-plane counts*; counts of different columns just add. So a neuron whose inputs live in several tiles is `fan` columns, one per tile, and
the mask lights all of them at once; the tree adds the counts of all of them (`S[a][b] = sum_c cnt[c][a][b]`) BEFORE the digit fold / shift /
relu / clamp. The partial sum is exact int and un-clamped by construction. A tile's rows are written only by that tile's `ACT` tokens (write gate
`beta = [tile(token) == tile(column)]`, a token feature, zero cost), so the same 16 key rows are reused by every tile and K is not bounded by the
128 keys of a head, only by `fan`.
* extra layers: the tree gets wider digits. Measured (build only): u5s5 rows=16: 1 col 7 layers; fan 2: 10 layers (smax 0); with smax=4: 13.
* limits: counts must stay bf16-exact in the mux (`fan*rows <= 256`), and the carry staircase window of `nonlin.requant` (256) must fit the
  digit ranges: `fan*rows` up to ~48 at u5s5 (rows=32, fan=2 raises `ValueError` from `requant`; rows=16, fan=3 builds). Weights outside
  [-16,15] (from gather duplicates) are split over extra columns of the same tile, using up fan.
* what is NOT solved: a neuron needing more than `fan*(rows-1)` inputs (e.g. 784) would need digit-token partial sums across *time*: tile-1 activations
  overwritten by tile-2 tokens before a second pass, which requires either an activation RAM (re-emitting ACT tokens from state) or
  prompt-interleaved READs (host sequencing). Not built. Registers also bound the input width: all inputs and all activations are registers
  (SSA), `nreg = tiles*rows`, and each tile costs heads.

## 4. Problem 3: ops

* **relu_clamp**: the requant gadget includes relu (`clamp(floor(y/2^k),0,M) = clamp(relu(y)>>k,0,M)`).
* **shift 0..smax per op**: `y' = y * 2^(smax - s)` is formed by *selecting* the digit positions of the counts with the one-hot shift read from
  the program memory (`mux`: `sign * S[a][b] * onehot_s`, one FFN level, `AB*WB*(smax+1)` units), then a FIXED `requant(shift=smax)` runs once
  (`floor(y 2^(smax-s) / 2^smax) = floor(y/2^s)`). Digits are per position `p = a+b` (weight 2^p, sign of the top weight bit), not per
  activation plane, which keeps every digit's range at `~5*fan*rows` and the carry staircase inside the 256 gate window. Exact against numpy on
  random programs and on worst cases (all +15/-16 weights, all-31 activations, shifts 0..7), u3s3 and u5s5.
* **residual add**: IR.md has no ADD; the frontend emits identity rows (weight `2^shift`) over `concat`, summed in the tree: free.
* **bias_rows = b**: `b` constant-1 registers per tile, bias split over their rows of the neuron's first column.
* **gather / concat / flatten**: scheduler re-indexes (tensor = list of register references); duplicated source registers sum their weights; sums
  outside the weight range are split over columns. `groups=P` (conv positions): the weights are *duplicated* per position (P x the neurons):
  a conv layer costs P times the columns.

## 5. Layer / size cost (measured on the built graphs, `g.n_layers()`)

| machine | chip layers |
|---|---|
| qwenTPU v2 baseline 16x16 u5s5 / 32x32 u5s5 | 10 / 20 |
| this: rows=16, 1 tile of 16 cols, smax 0 | 7 |
| this: rows=16, cols=(15,10,0), fan 1, smax 4 (the digits MLP machine) | **9** |
| this: rows=16, cols=(15,12,8), fan 2, smax 7, 40 ops (general machine) | ~14 |
| this: rows=32, 32 cols, fan 1, smax 7 | 14 |
| u3s3 rows=8, 8 cols | 3 (shift mux smax 3: 5) |

Heads: 2 per column + ~1 + fan + (1 shift) + ceil((nops+nreg)/128); 16 key heads per layer. Vocabulary 10K..25K tokens.

## 6. Problem 4: scheduler and generator

`sched.compile_graph(g, rows=16, spec_override=None)`: sizes a `Spec` for the graph (or checks that the graph fits a given machine:
`DoesNotFit` names what is missing). `Compiled.program(xq)` builds the prompt for one encoded input; `Program.expected()` is the golden
token stream; `Compiled.decode(emitted)` reads the outputs off the emitted ACT tokens. `run.generate(feed, prompt, halt)` is the entire host
loop. Input encoding (`reference.encode_input`) and argmax over the output vector are host-side, as IR.md says.

## 7. Verification

CPU (qwentpu float64 evaluator, all exactness conditions asserted): `pytest tests/test_machine_*.py` (program memory, tiling, ops u3s3 and
u5s5 extremes, scheduler incl. concat/gather/groups/bias_rows=2, and the frontend's real quantized digits MLP 15-15-10 u5s5 with shifts 2 and
4: outputs equal `reference.run`).

GPU (A100, real merged Qwen3.5-0.8B, bf16, host = `generate()` = argmax; append only): `scripts/gpu_run.py digits` ran the quantized digits MLP
15-15-10 (u5s5, shifts 2 and 4, rows=16 cols=(15,10,0) smax=4, **9 chip layers**, hidden 3072, vocab 17616, 22 s build) on 20 held-out digits with
both readouts (code bits and lm_head): **40/40 exact** (emitted ids == numpy token golden AND outputs == `reference.run`); prompt 434 tokens, 50
emitted; 95% argmax accuracy on those 20. Log: `~/qwentensor_run/digits.log` on the box.
`general` phase: ONE compiled 14-layer model (rows=16, cols=(15,12,8), fan 2, smax 7, 40 ops, hidden 3712, vocab 26616) ran four different programs, 6 inputs each, all **exact (24/24)**: digits MLP 15-15-10; random MLP 20-8-4 (K=20 spans 2 tiles); random 4-layer 8-5-5-5-4; residual block (concat + identity rows). Only the prompt differed.

## 8. Remaining unsolved (summary)
loops / data-dependent branches / early exit; neurons with more than `fan*(rows-1)` inputs (no activation RAM, no digit-token carry across
time); weight streaming (all weights resident: cells = sum over neurons of tiles-touched * rows); conv `groups` unrolled (P x columns); SSA register
file (program length <= registers); signed activations / `act=linear` (outputs are relu-clamped); attention / softmax / norms (IR level);
hardware size: heads ~ 2 per column (16 per layer), so ~100 neurons is the practical ceiling on this model.
