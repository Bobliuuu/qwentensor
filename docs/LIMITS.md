# Limits: how big can qwentensor get, and why

Measured on `main` at `ad2bc28`, the circuit before the wide-layer and vectorisation work, which are now merged: the inputs-per-neuron figures below are for ONE adder tree and are superseded for wide neurons by tile groups (section 8). Qwen3.5 text towers with the qwenTPU splice, bf16.
Every row says how it was established:

* **[real]** ran on the real merged Qwen3.5 on the A100 (log in `results/`).
* **[sim]** ran on the exact float64 circuit evaluator (every bf16 / silu / state-row exactness condition asserted), CPU.
* **[build]** the circuit builds (or its own assertion rejects it); this is necessary but not sufficient for exactness (section 3.2).
* **[derived]** follows from the code and the configs; no run.

Probes: `scripts/probe_limits.py <probe>` (CPU) and `scripts/probe_limits_gpu.py <case>` (A100). Boundary tests: `tests/test_limits.py`.
Logs: `results/limits_*.log`, `results/a100_limits_*.log`.

## 1. The answer

One machine, one run, on the real Qwen3.5-0.8B, with **every column and every op slot the circuit has in use**:

| | value | how |
|---|---|---|
| neurons per machine | **126** | 126 physical columns; 128 op slots |
| depth inside one run | **126** sequential neurons | a chain of 126 one-input neurons [real], see section 4 |
| inputs per neuron, one adder tree | **63** at any shift 1..4, **49** at shift 5..6, **45** at shift >= 7; **254** with no shift (the tree is capped at `fan * rows <= 48` in the code, `spec.GROUP_F`) | section 3 [sim] |
| inputs per neuron, tile groups | `G x 45` at rows 16 (`G` groups of 3 tiles, [WIDE.md](WIDE.md)); ran on the real 0.8B with 64, 96, 512 and 784 inputs (`results/a100_wide*.log`); the upper bound from tiles, mask heads and the 126-column mask is **not re-measured** | section 8 [real] |
| weights per machine | ran: **4,335** placed (rows 64, 83 columns x 63 inputs) and 3,208 (rows 32, 124 columns) [real]. Ceiling `126 x (rows - 1)`: 3,906 at rows 32, 7,938 at rows 64 (needs about 44 GB), 16,002 at rows 128 (shift 0 only, about 54 GB) | section 4 [real]; the ceilings are [derived] from `results/limits_rows.log` |
| activation / weight bits | activations <= **9** bits, weights >= 16 bits arithmetic; **5 / 5 is the widest at which 126 columns fit one A100-40GB** (6 / 6 needs about 46 GB) | section 5 |
| network size | **unbounded**, by running pages on one machine (host copies the integer activations); each neuron has at most `G x 45` inputs with tile groups (single tree: 63, or 254 at shift 0) | section 6 |

Two different walls set these:

1. **The 128-wide head.** Every Qwen3.5 size has `linear_key_head_dim = linear_value_head_dim = 128`. A program-memory head stores one 128-wide word per
   op slot (the key is the slot, one-hot, so at most 128 slots) and the word carries a 2-column reference ballast plus one bit per physical column, so at most
   126 columns. That is why one machine has **126 neurons and 128 op slots on every model size**: [derived] from the HF configs for 2B to 27B, [real] on 0.8B only (the 2B and 4B models were run at 35 columns, not 126).
2. **bf16's 8-bit significand.** The datapath is bf16 and integers are exact only up to 256 (`bf16(257) = 256`). Every bit-plane count, every digit of the
   accumulator and every threshold of the requantiser must be an integer <= 256. That caps the accumulator of one neuron (a sum of `fan * rows` products) and,
   with the shift, how many inputs it can have.

Nothing else is a wall: memory and time grow with the machine but a bigger GPU or more time moves them.

## 2. Hard bounds, one by one

### 2.1 Head geometry: columns, op slots

| bound | value | cause | evidence |
|---|---|---|---|
| physical columns per machine | 126 | `circuit.mem_head`: `assert nb <= 126`. The mask head's value word is 128 columns; columns 0 and 1 are the reference ballast (`A_REF`), one column bit per physical column. | build with 126 columns ok, 127 raises `AssertionError` [build] (`results/limits_size.log`); a 126-column machine ran exact on the real model [real] |
| op slots per machine | 128 | `assert NOPS <= 128`: the mask / shift head keys are the one-hot op slot, 128 channels. The successor heads already split beyond 128 (`nsucc_heads`); mask and shift do not. | 128 ok, 129 raises [build] |
| neurons per machine | min(columns, ops) = **126** | a neuron needs >= 1 column and 1 op | [real] (section 4) |
| registers | not capped by heads; capped by vocabulary | `ACT(r;v)` is one token per register and value, `PS_READ`/`PS_ACT` are `nops * nreg` tokens each. Tiles reuse the same 16 accumulator key rows (write gate = tile). | nvocab 85,988 at 126 columns on one tile, 126,776 at 160 registers [build] |
| heads and layers | `2 * columns + ceil((ops + registers) / 128) + fan + 1` key heads, 16 per chip layer | one weight head `W[c]` and one accumulator head `A[c]` per column, plus the successor, mask and shift heads. Each uses one of the 16 key heads (and one value head) of a layer, so layers grow with columns: 8 at 15 columns, 11 at 45, 18 at 100, 21 at 126 | `results/limits_size.log` [build]: 256 heads at 126 columns. Not a wall: depth costs memory (section 2.4). |
| accumulator value channels | `ab * wb <= 126` | one head value column per (activation bit, weight bit) count, columns 0 and 1 reserved | [derived] (the activation ceiling below is reached first) |

Unbuilt, not impossible: more than 126 columns per machine or 128 ops would need several mask / shift heads (the successor memory already does this),
at one more head per 126 columns.

### 2.2 Arithmetic: `fan * rows`, the shift, and what "builds" hides

One neuron sums `fan` columns, each holding `rows` weights: `F = fan * rows` products per bit plane. The count of one (activation bit, weight bit) plane is
`S[a][b] <= F`. The circuit forms digits of the accumulator by position `p = a + b` and a one-hot shift mux, then runs the requantiser (`nonlin.requant`).

| bound | value | cause | evidence |
|---|---|---|---|
| plane counts | F <= 256 | `assert F <= 256` ("sum of fan*rows counts must stay bf16-exact") | `bf16(257) = 256` (`results/limits_bf16.log`) [build] |
| requantiser window | F <= 256 / 256 / 192 / 120 / 80 / 56 / 48 for smax = 0 / 1 / 2 / 3 / 4 / 5-6 / >= 7 (best over rows 4..128) | the carry staircase must fit the 256-wide gate window: `ValueError: carry staircase ... wider than the 256 gate window` | `results/limits_fan.log` (rows 4 to 128, smax 0 to 12) [build] |
| **shift mux digits** | **F <= 256 / min(ab, wb-1) = 64** when smax >= 1 | the mux materialises digit values up to `min(ab, wb-1) * F` as bf16 residual dims; odd integers above 256 are not representable | `fan*rows = 64` exact, 72 / 80 / 128 not (section 3.2) [sim] |

### 2.3 Bit widths

| bound | value | cause | evidence |
|---|---|---|---|
| activation bits | <= 9 | the bit-extraction thresholds `[u >= m 2^a]` need a gate window <= 256: ab = 10 raises `AssertionError: (257, 0, 1023)` | ab = wb = 9 exact [sim]; 10, 11, 12 rejected (`results/limits_bits_stress.log`) |
| weight bits | >= 16 arithmetic | counts stay `<= F`; weights are stored as bits | ab = 2, wb = 16 exact [sim] |
| what stops the bit width first | memory | `2^ab` ACT tokens per register and `2^wb` CELL tokens per weight cell grow the vocabulary, and `ab * wb` count channels per column grow the residual (hence hidden size) | section 5 |

### 2.4 Memory (what actually limits columns on a given GPU)

The splice widens Qwen's own layers to the chip's hidden size (zero padding stores them at full width), appends the chip layers and adds an untied
embedding and `lm_head` over `248,320 + nvocab` tokens:

`GB = 2 * ((L + chip_layers) * per_layer(hidden) + 2 * (248,320 + nvocab) * hidden) / 1e9`

Calibration: this formula gives 30.6 GB for the 0.8B / 126-column machine; the real run built to **33.1 GB** (hidden 9,728 and 22 chip layers, a slightly
larger Spec than the probe's). See section 5 for the table.

### 2.5 Prompt, context, depth, time

| bound | value | evidence |
|---|---|---|
| prompt length | 2,424 tokens at 126 columns, rows 16; 4,860 at rows 64 (83 columns, 4,335 weights); 3,865 at rows 32 | [real] |
| emitted tokens | `2 * ops` = 252 at 126 ops (plus the HALT) | [real] |
| context window | 262,144 on every size | HF configs; the full machine uses 0.9% of it |
| depth | <= op slots in one run (every neuron is one READ/ACT pair), unbounded across pages | section 4 |
| wall time of the full machine, eager | about 8 s per inference (126 columns) | [real], section 4 |

## 3. Pushing each bound until it breaks

### 3.1 Build-time walls (`probe_limits.py size | fan | bits | rows`)

* `size`: 126 columns / 126 ops builds (21 layers, 8,365 residual dims, 256 key heads, 3,692 FFN units); 127 columns or 129 ops raises `AssertionError`.
* `fan`: for each (rows in 4..128, shift cap in 0..12) the largest `fan` that builds, with the first failure message (`results/limits_fan.log`).
* `bits`: all (ab, wb) in 1..9 build at rows 16; `ab = 10, 11, 12` raise (`results/limits_bits.log`, `results/limits_bits_stress.log`).
* `rows`: 126 columns at rows 8, 16, 32, 64, 128 (`results/limits_rows.log`): rows 64 builds only at shift <= 4 and rows 128 only at shift 0.

### 3.2 A build is not exactness (`probe_limits.py stress | extremes`)

The circuit's own checks accept more than it can compute exactly. With all activations at `amax` and weights near `wmax` with random cleared bits, the
exact evaluator asserts that every bf16 value equals its exact integer. Results (`results/limits_stress*.log`, `results/limits_extremes.log`):

| rows | fan | smax | F = fan*rows | builds | exact |
|---|---|---|---|---|---|
| 16 | 3 | 7 | 48 | yes | 16/16 |
| 8 | 7 | 5 | 56 | yes | 16/16 |
| 16 | 4 | 1 | 64 | yes | 16/16 |
| 64 | 1 | 1, 2, 3, 4 | 64 | yes | 16/16 each |
| 32 | 2 | 2, 3, 4 | 64 | yes | 16/16 each |
| 4 | 16 | 1 | 64 | yes | 16/16 |
| **4** | **17** | **1** | **68** | yes | **14/16** |
| **8** | **9** | **1, 3** | **72** | yes | **13/16** |
| **16** | **5** | **1, 2, 3** (15/20 at 4) | **80** | yes | **12/16** |
| **64** | **2** | **1** | **128** | yes | **7/16** |
| 16, 128, 32 | 16, 2, 8 | 0 | 256 | yes | 8/8 each |
| 16, 128 | 17, 3 | 0 | 272, 384 | no | n/a |

The failure is `mux4: bf16 output differs from exact: bad value 272.0`: a digit of 4 planes x 72 = up to 288 whose odd values do not survive bf16. With no
shift (smax = 0) the circuit has no mux and F = 256 is exact. **`probe_fan`-style "it builds" bounds above 64 (the 80 in `docs/STREAMING.md` section 3.2)
are not exact bounds.** The test `tests/test_limits.py::test_fan_times_rows_64_is_exact_and_80_is_not` pins this boundary.

Verified-exact maximum inputs per neuron, `fan * (rows - 1)` (one constant row per tile is the bias):

| shift cap smax | largest exact F | best (rows, fan) | inputs per neuron |
|---|---|---|---|
| 0 | 256 | (128, 2) | 254 |
| 1 to 4 | 64 | (64, 1) | 63 |
| 5, 6 | 56 | (8, 7) | 49 |
| >= 7 | 48 | (16, 3) | 45 |

## 4. The full machine on the real model

`scripts/probe_limits_gpu.py` (cases below), A100-40GB, bf16, host = `generate()` = `argmax; append`. A case passes when the emitted tokens equal the numpy
token golden and the outputs equal `reference.run`.

All four cases on Qwen3.5-0.8B, 6 random inputs each, code readout, **24/24 exact, 0 mismatches**:

| case | machine | what it fills | chip layers / hidden / vocab | prompt, emitted | peak GPU | log |
|---|---|---|---|---|---|---|
| `wide126` | rows 16, cols (15 x 8, 6), smax 3 | **126 neurons, 126 columns**, 1,759 weights | 22 / 9,728 / 126,776 | 2,424 / 252 | 33.4 GB | `results/a100_limits_wide126_q08.log` |
| `deep126` | rows 16, cols (15 x 8, 6), smax 0 | **depth 126**: a chain of 126 neurons, 126 sequential READs | 20 / 9,728 / 121,712 | 744 / 252 | 32.2 GB | `results/a100_limits_deep126_q08.log` |
| `rows32` | rows 32, cols (31 x 4), smax 3 | 124 neurons x **31 inputs**, 3,208 weights placed | 24 / 9,728 / 188,088 | 3,865 / 248 | 36.8 GB | `results/a100_limits_rows32_q08.log` |
| `wide63` | rows 64, cols (63, 20), smax 4 | 83 neurons x **63 inputs** (the widest exact neuron with a shift), 4,335 weights placed | 26 / 7,424 / 215,694 | 4,860 / 166 | 29.8 GB | `results/a100_limits_wide63_q08.log` |

Per-page time of the full machine, eager chip path on a shared A100: `wide126` 49 s for 6 inputs, about 8 s per inference (2,424 prefill + 252 emitted tokens).
Build (CPU splice) took 175 to 270 s per model. The 2B and 4B models were not run at 126 columns: the estimate of section 5 is 41 GB and 66 GB, above the 40 GB card.

## 5. Per model size

HF `text_config` of each size: `linear_key_head_dim = linear_value_head_dim = 128` and `linear_num_key_heads = 16` on **all** of them, so every bound in
section 2.1 to 2.3 is the same for every size. What differs is the host model's width and depth, hence the memory.

| model | hidden | layers | key heads | value heads | intermediate | columns per machine | est. GB at 126 columns | max columns in 40 GB / 80 GB | largest machine run |
|---|---|---|---|---|---|---|---|---|---|
| 0.8B | 1,024 | 24 | 16 | 16 | 3,584 | 126 | 30.6 (33.1 measured) | 126 / 126 | 126 [real] |
| 2B | 2,048 | 24 | 16 | 16 | 6,144 | 126 | 41.2 | 105 / 126 | 35 columns [real] (`results/a100_general_q2.log`) |
| 4B | 2,560 | 32 | 16 | 32 | 9,216 | 126 | 66.2 | 60 / 126 | 35 columns [real] |
| 9B | 4,096 | 32 | 16 | 32 | 12,288 | 126 | 87.7 | 30 / 90 | none |
| 27B | 5,120 | 64 | 16 | 48 | 17,408 | 126 | 190.8 | 0 / 0 | none |

The "est. GB" and "max columns" columns are [derived] from `probe_limits.py config` (builds the circuit at 15 to 126 columns and applies the formula of
section 2.4, with 8% headroom). The MoE sizes (35B-A3B, 122B-A10B, 397B-A17B) have no dense `intermediate_size`; the splice in `qwentpu/merge.py` handles dense
MLP tensors only, so they are not covered.

Bit width and memory at the full machine (`probe_limits.py bitmem`, 0.8B, estimated GB):

| ab = wb | residual dims | chip layers | vocabulary | 0.8B at 45 columns | 0.8B at 126 columns |
|---|---|---|---|---|---|
| 3 | 4,065 | 20 | 37,196 | 6.6 GB | 15.3 GB |
| 4 | 5,961 | 20 | 53,460 | 8.3 GB | 21.9 GB |
| 5 | 8,366 | 21 | 85,988 | 11.2 GB | 30.6 GB |
| 6 | 11,278 | 25 | 151,044 | 15.3 GB | 45.9 GB |
| 7 | 14,692 | 27 | 281,156 | 20.7 GB | 67.7 GB |
| 8 | 18,611 | 30 | 541,380 | 28.7 GB | 111.2 GB |

On an A100-40GB the full 126-column machine fits at ab = wb = 5 (the 30.6 GB estimate, 33.1 GB measured) and not at 6; at 45 columns it fits up to 8 bits.
These are [derived] from the build plus the section 2.4 formula; only 5 bits was run on the real model.

## 6. Paged execution

A network larger than one machine runs as pages (`docs/STREAMING.md`): each page is a fresh prompt on the same merged model, the host copies the integer
activations (`0..amax`) from one page into the next.

| limit | value | why |
|---|---|---|
| neurons in a network | unbounded | pages are independent runs; nothing accumulates in the model between pages |
| neurons per page | <= 126 | section 2.1 |
| inputs per neuron | the section 3.2 table | the paging does not change one neuron's accumulator |
| prompt per page | <= rows * columns + 4 * ops + registers ~ 2.5K (rows 16), 17.6K (rows 128) tokens, vs 262,144 | [derived]; 2,424 measured at 126 columns |
| time | `pages x (prefill + 2 * ops + 1 emitted tokens)` | [real] section 4 gives the per-page time of the full machine |
| host involvement | the page sequence and the activation copies | weaker than qwenTPU v2's model-driven loop |

## 7. Impossible in principle, versus unbuilt

Nothing in this list is impossible in the architecture; the two walls of section 1 are encodings (head width 128, bf16 significand), and each has a
workaround that costs more heads or more bits. What is checked and what is argued:

| item | status | argument |
|---|---|---|
| MaxPool | **buildable today with no circuit change** [sim] | `max(a, b) = b + relu(a - b)`: two matvec layers, three neurons for one pair. All 1,024 pairs equal on the numpy golden, 9/9 on the circuit (`results/limits_maxpool.log`). Only the frontend lowering is missing. |
| signed outputs | **buildable today** [sim] | offset binary: `out = clip(y + 16, 0, 31)` with two constant rows; all 1,024 (a, b) pairs equal `clip(a - b, -16, 15) + 16` on the golden, 6/6 on the circuit (`results/limits_signed.log`). The quantiser does not emit it. |
| pointwise nonlinearities (GELU, SiLU, sigmoid, tanh, exp) | unbuilt, **no new mechanism** [derived] | a function of a 5-bit value is a 32-entry table, a sum of <= 31 relu steps (`docs/IR.md` section 3): ~32 neurons per scalar, so the 126 ops of one machine are the budget, not the circuit |
| loops, data-dependent branches, early exit | unbuilt [derived] | a successor is `succ[slot(current token)]` (key = slot). Keying it on (slot, value) needs 32x the key channels, i.e. several 128-wide heads. Reusing a register in a loop is fine: re-issuing `ACT(r; v)` overwrites its accumulator row (`scripts/probe_stale.py`, `docs/STREAMING.md` section 2); what is missing is a counter and a value-keyed exit. |
| data x data products (attention, LayerNorm scale, softmax division) | unbuilt, needs a new mechanism [derived] | the array multiplies two bit-planes by AND-counting, so a runtime-written operand would work; today only `CELL` tokens write the weight heads. An `ACT` that also writes a weight row needs a column address from the program memory. |
| a neuron with more than the section 3.2 inputs | impossible **in bf16 with this encoding** | integer planes cannot exceed 256 in bf16. Running the merged model in fp32 (24-bit significand) or fp16 (11-bit) would move the 256 to 2^24 / 2^11 at 2x / 1x the memory of bf16 weights; not tried. Partial sums across neurons need un-clamped digits carried between ops: unbuilt. |
| more than 126 columns / 128 ops per machine | unbuilt | more mask / shift heads (section 2.1) |
| unbounded working memory in one run | **impossible** here | every chip layer is Gated DeltaNet: the working memory is the fixed-size head state, and registers are limited by vocabulary. Random access to earlier tokens needs full attention, which the chip layers do not use. Larger state means pages and the host. |

## 8. What the merged work moved

Everything above was measured on `main` at `ad2bc28`. Two changes were merged afterwards:

* **Wide layers** (`docs/WIDE.md`, tile groups). Neurons reading more than 48 counts of one adder tree are split into groups of 3 tiles whose carry chains are normalised
  before the groups are added. The code never runs a single tree above `fan * rows = 48` (`spec.GROUP_F`), which sits below the measured exact ceiling of 64 in section 3.2.
  On the real 0.8B the 64-wide residual MLP, the 64-96-64-32-10 deep MLP, the CNN's 512-wide Linear and a 784-wide first layer ran exact in pages
  (`results/a100_wide_residual.log`, `a100_wide_deep.log`, `a100_wide_cnn.log`, `a100_wide784.log`). This replaces the inputs-per-neuron rows of sections 1 and 3.2 for
  wide neurons. It does not move section 2.1 (126 columns, 128 ops, heads) or section 5 (memory per model). The new ceiling of the grouped path (`G` bounded by the tiles,
  the `fan` mask heads and the 126-column mask value) was not probed to failure.
* **Vectorisation** (`docs/VECTORIZED.md`) emits several neurons per token: it moves the emitted-token count and time, and trades against the vocabulary
  (`nout^k` token ids) and the code bits; it does not change the 126-column wall. It is not combined with tile groups: `qt.compile(..., lanes=N)` raises `DoesNotFit` for a neuron wider than one adder tree.
* `tests/test_limits.py` is part of the suite and passes on the merged `main`; it fails on purpose if a bound moves.

## 9. Reproduce

```
python scripts/probe_limits.py size | fan | bits | rows | bf16 | config | bitmem | extremes | stress rows,fan,smax[,ab,wb]... | maxpool | signed
bash scripts/box/_limits_batch2.sh       # A100 box: wide63 and rows32 of section 4 (wide126 and deep126: `python scripts/probe_limits_gpu.py wide126 | deep126`)
pytest tests/test_limits.py
```
