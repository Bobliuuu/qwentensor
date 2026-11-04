"""Stage 3: ops.  Per-op requant shift (exact floor, relu, clamp) selected by the program memory; wide sums from K-tiling."""
import numpy as np

from qwentensor.machine.asm import Op, Program
from qwentensor.machine.run import IRMachine
from qwentensor.machine.spec import Spec
from test_machine_prog import check


def rand_prog(sp, rng, nops=7, wide=True):
    W = {}
    for c in range(sp.ncols):
        for s in range(sp.rows):
            if rng.random() < (0.8 if wide else 0.5): W[(s, c)] = int(rng.integers(sp.wlo, sp.whi + 1))
    init = {r: int(rng.integers(0, sp.amax + 1)) for r in range(2 * sp.rows)}
    init[sp.rows - 1] = 1
    ops, order = {}, []
    for i in range(nops):
        k = int(rng.integers(1, sp.fan + 1))
        cols = [int(c) for c in rng.choice(sp.ncols, k, replace=False)]
        ops[i] = Op(cols, int(rng.integers(0, sp.smax + 1)), 2 * sp.rows + i); order.append(i)
    return Program(sp, W, ops, order, init, entry=sp.rows - 1), W, init


def test_shift_and_wide_sums():
    sp = Spec(rows=8, cols=(4, 4, 0), ab=3, wb=3, nops=8, fan=2, smax=3)
    m = IRMachine(sp); print(m.stats())
    rng = np.random.default_rng(11)
    shifts = set(); sat = set()
    for trial in range(3):
        prog, W, init = rand_prog(sp, rng)
        p, em = check(m, prog)
        for i, (j, o) in enumerate(prog.ops.items()):
            y = 0
            for c in o.cols:
                t = sp.col_tile(c)
                y += sum(W.get((s, c), 0) * init[t * 8 + s] for s in range(8))
            want = int(np.clip(y >> o.shift, 0, sp.amax))
            assert em[2 * i + 1] == sp.act(o.dest, want)
            shifts.add(o.shift); sat.add("neg" if y < 0 else ("hi" if (y >> o.shift) > sp.amax else "mid"))
    assert shifts == {0, 1, 2, 3} and sat == {"neg", "hi", "mid"}, (shifts, sat)


def test_u5s5_extremes_and_random():
    """Real number format (5-bit activations, 5-bit weights), rows=16, fan=2, shifts 0..7, worst-case sums included."""
    sp = Spec(rows=16, cols=(6, 6, 0), ab=5, wb=5, nops=8, fan=2, smax=7)
    m = IRMachine(sp); print(m.stats())
    rng = np.random.default_rng(21)
    prog, W, init = rand_prog(sp, rng, nops=6)
    # worst cases: column 0 all +15 / column 6 all -16 with max activations, shift 0 and 7
    for s in range(16): W[(s, 0)] = 15; W[(s, 6)] = -16; W[(s, 1)] = 15; W[(s, 7)] = 15
    for r in range(32): init[r] = 31
    init[15] = 31
    prog.W = W; prog.init = init
    prog.ops[0].cols, prog.ops[0].shift = [0, 1], 0          # +15*31*15 + ... huge positive -> clamp
    prog.ops[1].cols, prog.ops[1].shift = [6], 7             # huge negative -> 0
    prog.ops[2].cols, prog.ops[2].shift = [0, 1], 7          # (2*16*15*31)>>7 = 116 -> clamp 31
    prog.ops[3].cols, prog.ops[3].shift = [6, 7], 7          # mixed
    check(m, prog)
