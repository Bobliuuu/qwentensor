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


def test_paged_784_wide_first_layer():
    """A 784-wide first layer (53 tiles, 18 groups) runs paged, bit-exact in the circuit simulator."""
    from qwentensor.machine.pages import compile_paged, golden_executor, sim_executor
    from test_machine_stream import mk_mlp, run_paged
    rng = np.random.default_rng(11); g = mk_mlp(rng, [784, 6, 4], [6, 5], ab=5, wb=5)
    for n in g.matvecs(): n.W = rng.integers(1, 4, n.W.shape) * rng.choice([-1, 1], n.W.shape)   # small nonzero weights: every register is read
    pc = compile_paged(g, rows=16, max_cols=60, max_ops=8)
    assert pc.spec.ngroups == 18 and pc.spec.fan == 53
    xq = rng.integers(0, 32, (4, 784))
    run_paged(pc, g, xq, golden_executor())
    run_paged(pc, g, xq[:1], sim_executor(pc.spec))


def test_group_overflow_is_reported():
    """Duplicated source registers sum their weights, and a sum beyond the 5-bit range is split over extra columns of the same tile.
    Too many columns inside one tile group is a DoesNotFit, not a wrong answer."""
    import pytest
    from qwentensor.ir import Graph, Node
    from qwentensor.machine.sched import compile_graph, DoesNotFit
    idx = list(range(64)) + list(range(15))                               # the first 15 inputs appear twice: weight 15 + 15 = 30 = two columns
    g = Graph([Node("x", "input", [], 64), Node("d", "gather", ["x"], len(idx), index=np.array(idx)),
               Node("fc", "matvec", ["d"], 2, W=np.full((len(idx), 2), 15, dtype=np.int64), b=np.zeros(2, dtype=np.int64), shift=6)],
              "fc", ab=5, wb=5, profile="v2.1")
    with pytest.raises(DoesNotFit, match="tile group"): compile_graph(g, rows=16)


def test_single_tree_never_exceeds_the_exact_budget():
    """A neuron of 45 inputs whose weights need split columns (fan 4 at 16 rows = 64 counts) must not run on one adder tree: the shift mux of such a tree
    can produce bf16 values it cannot represent (docs/WIDE.md section 5).  It is routed to tile groups and, with 4 columns in one group, reported."""
    import pytest
    from qwentensor.ir import Graph, Node
    from qwentensor.machine.sched import compile_graph, DoesNotFit
    idx = list(range(45)) + list(range(15))
    g = Graph([Node("x", "input", [], 45), Node("d", "gather", ["x"], len(idx), index=np.array(idx)),
               Node("fc", "matvec", ["d"], 1, W=np.full((len(idx), 1), 15, dtype=np.int64), b=np.zeros(1, dtype=np.int64), shift=4)],
              "fc", ab=5, wb=5, profile="v2.1")
    with pytest.raises(DoesNotFit, match="tile group"): compile_graph(g, rows=16)
