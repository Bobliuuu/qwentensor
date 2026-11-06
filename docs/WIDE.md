# Wide neurons: more than 45 inputs per neuron

## 1. The limit and its root cause

One op sums the accumulator counts of up to `fan` columns (one column per touched tile, `rows` rows each) in one adder tree, then shifts, relus
and clamps. A neuron therefore reads at most `fan * (rows - 1)` registers. With `rows = 16` and shifts up to 5 the circuit builds only up to
`fan = 3` (45 inputs); `fan = 4` raises

    carry staircase for bit 4 spans [-64,452]: wider than the 256 gate window

(`scripts/probe_fan.py`; with `smax <= 4`, `fan = 5` builds and `fan = 6` fails the same way at bit 3, span [0, 588]). That a circuit BUILDS is not the
same as it being exact, see section 5: the exact ceiling of one tree is lower than the build ceiling.

Why: qwenTPU's gadgets compare an integer `z` with a threshold `theta` as `relu(z - theta + 1) - relu(z - theta)`. That pair is exact only if the
positive side stays within bf16's exact-integer range, `hi - theta + 1 <= 256` (or the mirrored form `theta - lo <= 256`; `Circ.terms_ge`). So a
staircase `floor(t / 2^q)` over an integer `t in [tlo, thi]` is realisable only if every threshold lies within 256 of one end of the range:
`thi - tlo` is at most about 512. The carry chain of the requantizer has `t = c + d_j`, where `d_j` is the digit at bit position `j` (a signed sum
of up to 5 bit-plane counts, each at most `F = fan * rows`) and `c` is the incoming carry. The carry range is about half of `t`'s range, so the
fixed point is `width(t) ~ 2 * width(d_j) ~ 10 F`. `10 F <= 512` gives `F <= ~48`, i.e. `fan * rows <= 48`. Raising `rows` does not help (`F` is
the product), a wider window would need an exact integer above 256 in bf16, which does not exist.

So the limit is not a count of columns or heads. It is the width of one exact carry chain over one adder tree.

## 2. The fix: groups with per-group carry normalisation

Partition the tiles into groups of `tpg` tiles (group `gamma` = tiles `gamma*tpg .. (gamma+1)*tpg - 1`). A column's group is the group of its tile, so
the partition is static and costs no extra heads. An op still names up to `fan` columns in the program memory (`fan` = total active columns, as before),
but at most `gfan` of them may lie in one group, with `gfan * rows <= 48`.

For each group the tree forms its own exact integer

    y_g = sum over the group's masked columns of (activation . weight)          (so |y_g| < 2^15)

as carry-save digits `d_p` (position `p = a + b`, `p = 0 .. 8` for 5-bit activations and weights), each `d_p` in a range of width at most `5 * gfan * rows`.
Then the carry chain `c_0 = 0`, `T_j = c_j + sum_{i<q} 2^i d_{j+i}`, `c_{j+q} = floor(T_j / 2^q)`, `L_j = T_j - 2^q c_{j+q}` runs over **all**
positions, and past the last digit (`d = 0`) until the carry range collapses to `[-1, 0]`. `q` is the largest step (at most 4) for which the
staircase of `T_j` passes the window test, exactly as in `nonlin.requant`, so every step is a legal exact stage.

Exactness: `floor` and `L = T mod 2^q` are exact integer identities, so

    y_g = sum_j 2^j L_j + 2^J c_J            with L_j in [0, 2^q - 1] and c_J in {-1, 0}.

All groups have the same ranges, hence the same `q` schedule. Summing over groups position by position gives the digit list

    D_j = sum_g L_{g,j} in [0, G (2^q - 1)],      D_J = sum_g c_{g,J} in [-G, 0],      y = sum_j 2^j D_j,

where `G` is the number of groups. These digits are tiny (`G (2^q - 1)` is far below 256 for `G <= 36`), so the existing one-hot shift mux and the
existing `nonlin.requant` (relu, floor shift, clamp, output bits) run on them unchanged, with the window test passing at any `G`. The stage is bit-exact because
every step is an integer identity over values that the window test certifies as exactly representable, and `circuit-IR` asserts that for every
node (the CPU simulator checks every bf16, silu and one-hot condition).

What this buys: a neuron may read `G * gfan * (rows - 1)` registers, with `G` bounded only by the tiles, the `fan` mask heads (one program-memory head per
active column) and the 126-column mask value width. At `rows = 16`, `tpg = gfan = 3` that is 45 inputs per group: 64 inputs need 2 groups, 96 need 3, 512
need 12, 784 need 18.

## 3. What it costs

* Chip layers: the per-group carry chain adds about `J / q` FFN levels (`J ~ 12`) before the shift mux; the legacy single-group path is unchanged and
  is selected when the machine has one group (`tpg = 0`).
* Heads and hidden size: no new heads (the partition is by tile). The residual stream gains the carry dims of the chain (a few per group per stage).
* Prompt size: unchanged per neuron (one CELL token per nonzero weight); a neuron spanning `t` tiles uses `t` columns and `t` mask tokens.
* Vocabulary: `PM` tokens grow with `fan * nops * (ncols + 1)`.

## 4. What it does not solve

* The columns per page: a 784-wide neuron needs 53 columns (one per touched tile), so a page holds few such neurons and the layer runs as many pages
  (host-driven paging, `docs/STREAMING.md`). The program memory mask value has 126 dims, so one machine has at most 126 columns.
* In-model weight streaming, loops and branches are unchanged.

## 5. A circuit that builds is not necessarily exact

The build-time window test only certifies the carry staircases. The shift mux (one-hot shift times a bit-plane count) hands digits of up to `5 * F` to the
requantizer as residual-stream values, and bf16 represents integers exactly only up to 256 (even integers above). Fork D (maximum-capacity probes,
reported to this branch by the coordinator; its probes are in its own branch and were not rerun here) measured the single-tree path against the exact evaluator:

| `fan * rows` | 64 | 68 | 72 | 80 |
|---|---|---|---|---|
| random programs exact | all | 14/16 | 13/16 | 12/16 |

so the exact ceiling of ONE adder tree is `fan * rows = 64`, not 80 (the earlier statement "up to 80 when `smax <= 4`" in `docs/STREAMING.md` and `docs/DESIGN.md`
described what builds). This branch therefore never runs a single tree above `GROUP_F = 48` (`spec.GROUP_F`): `sched.compile_neurons` routes any neuron with
`fan * rows > 48` (including fans caused by split columns) to tile groups, and `pages.fan_cap` caps the single-tree path at that budget. 48 is below the measured
64 on purpose: the per-group digits are at most `5 * 48 = 240` and every count is at most 48, both inside 256.

The grouped path has its own bf16 bound: the combined digits `D_j = sum_g L_{g,j}` are mux operands, so `G * (2^q - 1) <= 256` must hold. `wide.qmax_for(G)` lowers
the digit width `q` as `G` grows (q = 3 up to 18 groups, 2 up to 36, 1 up to 128) and `circuit.build` asserts the bound on every combined digit.
