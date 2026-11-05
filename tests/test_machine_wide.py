"""Wide neurons (docs/WIDE.md): a neuron reading more than gfan*(rows-1) registers, summed exactly over several tile groups."""
import numpy as np

from qwentensor.machine.asm import Op, Program
from qwentensor.machine.run import IRMachine
from qwentensor.machine.spec import Spec
from test_machine_prog import check


def _wide_program(sp, nin, nout, seed, extreme=False, shifts=(0,)):
    """nin inputs spread over tiles (rows-1 data rows per tile, the last row of every tile is the constant 1), nout neurons, one column per touched tile."""
    rng = np.random.default_rng(seed); dr = sp.rows - 1
    ntile_in = -(-nin // dr); W = {}; ops = {}
    regs = [(k // dr) * sp.rows + k % dr for k in range(nin)]
    for n in range(nout):
        cols = []
        for t in range(ntile_in):
            c = sp.col_id(t, n); cols.append(c)
            for s in range(dr):
                if t * dr + s < nin:
                    W[(s, c)] = int(rng.choice([sp.wlo, sp.whi])) if extreme else int(rng.integers(sp.wlo, sp.whi + 1))
            W[(dr, c)] = int(rng.integers(0, 3)) if t == 0 else 0
        ops[n] = Op(cols, int(shifts[n % len(shifts)]), (ntile_in + n // dr) * sp.rows + n % dr, f"n{n}")
    init = {r: (sp.amax if extreme else int(rng.integers(0, sp.amax + 1))) for r in regs}
    for t in range(ntile_in): init[t * sp.rows + dr] = 1
    return Program(sp, {k: w for k, w in W.items() if w}, ops, list(range(nout)), init, entry=dr), regs, W


def _expect(sp, prog, regs, W, nin, nout, ops):
    x = np.array([prog.init[r] for r in regs]); dr = sp.rows - 1; out = []
    for n in range(nout):
        y = 0
        for c in ops[n].cols:
            t = sp.col_tile(c)
            for s in range(sp.rows):
                y += W.get((s, c), 0) * prog.init.get(t * sp.rows + s, 0)
        out.append(int(np.clip(y >> ops[n].shift, 0, sp.amax)))
    return out


def test_two_groups_64_inputs():
    sp = Spec(rows=16, cols=(2, 2, 2, 2, 2, 1), ab=5, wb=5, nops=4, fan=5, smax=5, tpg=3, gfan=3)
    assert sp.ngroups == 2
    m = IRMachine(sp)
    for seed, ext, sh in ((1, False, (0, 3)), (2, True, (5, 0)), (3, True, (4, 2))):
        prog, regs, W = _wide_program(sp, 64, 2, seed, ext, sh)
        p, em = check(m, prog)
        want = _expect(sp, prog, regs, W, 64, 2, prog.ops)
        got = [sp.act_val(t) for t in em if sp.is_act(t) and sp.act_reg(t) in {o.dest for o in prog.ops.values()}]
        assert got == want, (seed, got, want)


def _run_wide(nin, nout, seed, extreme, shifts, smax=5):
    dr = 15; nt = -(-nin // dr)
    sp = Spec(rows=16, cols=(nout,) * nt + (nout,), ab=5, wb=5, nops=nout, fan=nt, smax=smax, tpg=3, gfan=3)
    m = IRMachine(sp)
    prog, regs, W = _wide_program(sp, nin, nout, seed, extreme, shifts)
    p, em = check(m, prog)
    want = _expect(sp, prog, regs, W, nin, nout, prog.ops)
    got = [sp.act_val(t) for t in em if sp.is_act(t) and sp.act_reg(t) in {o.dest for o in prog.ops.values()}]
    assert got == want, (nin, got, want)
    return sp, want


def test_three_groups_96_inputs_extremes():
    sp, want = _run_wide(96, 2, 4, True, (5, 1))
    assert sp.ngroups == 3


def test_twelve_groups_512_inputs():
    sp, want = _run_wide(512, 1, 5, False, (5,))
    assert sp.ngroups == 12 and sp.fan == 35
