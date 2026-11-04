# Weight streaming: running networks larger than one machine

Code: `qwentensor/machine/pages.py` (pager and host loop), `qwentensor/machine/sched.py` (`compile_neurons`), `qt.compile(..., paged=True)`.
Tests: `tests/test_machine_stream.py`. Probes behind the design: `scripts/probe_stale.py`, `scripts/probe_fan.py`. GPU: `scripts/gpu_run.py stream`.

## 1. The problem

One compiled machine holds every weight resident (`docs/EXECUTOR.md` section 8): columns are heads, about two heads per column, `CELL` tokens fill them
in the prompt, and a mask head holds at most 126 columns and 128 ops. A network with more neurons than one machine (the 202-neuron deep MLP, the
106-neuron residual MLP whose neurons each touch several register tiles) has to reuse the physical array over time. That is weight streaming: the
weights of the next group of neurons are written into the same columns, the way a TPU loads weight tiles from memory.

## 2. What the circuit does (probed, `scripts/probe_stale.py`, CPU circuit simulator)

| prompt | result |
|---|---|
| plain prompt, 3 inputs, one neuron | circuit == golden |
| `CELL(row 0, col 0, new w)` inserted **before** the input `ACT` tokens | the neuron uses the new weight (weights can be overwritten) |
| the same `CELL` inserted **after** the input `ACT` tokens | the neuron still uses the old weight: **products `act x weight` are baked into the accumulator row at `ACT` time** |
| `ACT(r; 0)` re-issued for a register after `ACT(r; 2)` | the accumulator row is overwritten: the old contribution is erased |

Consequences for any scheme that swaps weights inside one run:

1. A column's weights must already be the NEXT neuron's weights when that neuron's input registers are written, and the accumulator rows left by
   the previous use of that column are summed into the next READ (the mask query reads all rows). They must be erased first.
2. The only eraser is re-issuing `ACT(r; 0)` for the same register `r`. In the program memory a register's successor is a property of the register
   (SSA, `docs/EXECUTOR.md` section 2), so a register cannot be visited twice by the emitted chain. Erasing would need a counter or a loop, which the
   machine does not have.
3. The model could emit the new `CELL` tokens only through the same chain: each emitted token needs a successor word keyed by its own slot. One op
   per streamed weight (a READ whose successor class is CELL) is limited by the 128 op slots, so a run could stream at most about 128 weights: less
   than the resident columns it would replace. A per-position ROM (`STREAM(i)` tokens whose write key and value come from a head read) needs about
   `rows + cols + 2^wb` PROG tokens per weight in the vocabulary and pushes every head after the weight write one layer deeper.

These are design arguments built on the probe above and on the circuit code; the in-model variant is **not built**.

## 3. What is built: paged execution

The network is cut into **pages**. A page is a normal machine program (`sched.compile_neurons`): a set of output neurons of one `matvec` node,
the registers holding exactly the source elements they read, their weights as `CELL` tokens, their schedule as `PROG` tokens. All pages share
**one `Spec`** (one merged model, built once). Each page is a fresh run: a new prompt means fresh DeltaNet state, so no stale accumulator row exists.

```
for page in pages:                         # host
    prompt = CELL*(page weights) PROG*(page schedule) ACT*(page inputs, copied from the activation store) ACT(entry)
    emitted = generate(prompt)             # argmax; append until HALT, entirely in the model
    store[page outputs] = values of the emitted ACT tokens
```

What the host does: builds each prompt, copies integers `0..amax` from the activation store into the next page's `ACT` tokens (the same
encoding it already does for the network input, plus the `gather`/`concat` re-indexing the scheduler already does), reads the emitted `ACT`
tokens back. What it does **not** do: any multiply, add, shift, clamp or argmax inside a page. Between pages the activation vector lives in the host
(an activation store); inside a page everything is token-driven as before. This is weaker than qwenTPU v2's "model runs the whole program": the
page sequence is host-driven.

Exactness: a page is exactly the machine program that `tests/test_machine_*.py` and the A100 runs already verify, and its outputs are integers
`0..amax`, so they are carried between pages without loss. Each neuron computes `clip(floor((sum w x + bias) / 2^shift), 0, amax)` with the same
integers as `reference.run`, so the composition over pages equals `reference.run` by induction over the nodes. Residual adds (identity rows over
`concat`) and `gather` work unchanged: the skip tensor is just entries of the store.

### 3.1 Packing

For each `matvec` node the neurons (one per output element, per `groups` position) are cut greedily into the largest prefix that fits
`max_cols` physical columns and `max_ops` ops (found by bisection, monotone in the prefix). The page's input vector is the distinct source
elements its neurons read, ordered by first use (a conv neuron's receptive field stays in adjacent registers, so it touches few tiles). Pages are
compiled standalone first; the shared `Spec` is the elementwise maximum of the pages' needs (tiles, columns per tile, `fan`, `smax`), and every page is
recompiled against it. The circuit is built once as a feasibility check. `DoesNotFit` names the limit that failed.

### 3.2 Limits that streaming does not change

* **Wide neurons.** One neuron reads at most `fan * (rows - 1)` registers. The circuit builds `fan * rows` up to 48 when shifts reach 5 or more
  and up to 80 when `smax <= 4` (`scripts/probe_fan.py`): the carry staircase of the requantizer must fit its 256 window. A trained 64-input layer
  needs shifts of 5 to 6, so its neurons need `fan` 5 at 16 rows and the circuit does not build. The 64-96-64-32-10 deep MLP (fan 7 or 8) and the
  64-wide residual MLP (fan 5 or 6) therefore still raise `DoesNotFit` when paged (`scripts/probe_wide_examples.py`, with and without `bias_rows=4`),
  now for this reason and not for lack of ops or columns. Partial sums across pages would need un-clamped
  digit carries between pages; not built.
* No loops or data-dependent branches inside a page; signed logits; attention, softmax, norms (unchanged).
* Throughput: every page re-prefills its own prompt, so a network costs `pages x (prompt prefill + emitted tokens)`.

## 4. Verification

Filled in from the runs: see section 5 (CPU circuit simulator) and `results/a100_stream.log` (real Qwen3.5-0.8B).
