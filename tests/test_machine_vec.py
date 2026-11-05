"""Vectorized machine (machine/vec.py): CPU circuit-IR simulator vs the numpy token golden and reference.run."""
import numpy as np
import pytest

from qwentensor import reference as R
from qwentensor.ir import Graph, Node
from qwentensor.machine.vec import VecMachine, compile_vec, reference_emit


def mk_graph(rng, dims, shifts, ab=3, wb=3, bias_rows=1):
    nodes = [Node("x", "input", [], dims[0])]
    prev, plen = "x", dims[0]
    for i, (n, sh) in enumerate(zip(dims[1:], shifts)):
        lo, hi = -(1 << (wb - 1)), 1 << (wb - 1)
        nodes.append(Node(f"fc{i}", "matvec", [prev], n, W=rng.integers(lo, hi, (plen, n)), b=rng.integers(lo, hi, n), shift=sh))
        prev, plen = f"fc{i}", n
    return Graph(nodes, prev, ab=ab, wb=wb, profile="v2.1", bias_rows=bias_rows)


def check(comp, m, xq, g):
    prog = comp.program(xq); p = prog.prompt(); em = prog.expected()
    assert m.run(p, em) == em + [comp.spec.HALT], "circuit != numpy token golden"
    assert (comp.decode(em) == R.run(g, xq[None])[0]).all(), "decoded outputs != reference.run"
    return len(p), len(em)


@pytest.mark.parametrize("lanes", [1, 2])
def test_mlp_bit_exact(lanes):
    rng = np.random.default_rng(4)
    g = mk_graph(rng, [6, 5, 3], [1, 2])
    comp = compile_vec(g, rows=8, lanes=lanes)
    m = VecMachine(comp.spec)
    for _ in range(3):
        _, ne = check(comp, m, rng.integers(0, 8, 6), g)
    assert ne == -(-5 // lanes) + -(-3 // lanes)            # one token per op group (+ nothing else)


def test_token_count_vs_scalar():
    from qwentensor.machine.sched import compile_graph
    rng = np.random.default_rng(5)
    g = mk_graph(rng, [6, 6, 4], [1, 2])
    xq = rng.integers(0, 8, 6)
    scalar = len(compile_graph(g, rows=8).program(xq).expected())
    for lanes, want in ((1, 10), (2, 5)):
        comp = compile_vec(g, rows=8, lanes=lanes)
        assert len(comp.program(xq).expected()) == want < scalar == 20


def residual(rng, w=5, ab=3, wb=3):
    nodes = [Node("x", "input", [], w), Node("fc0", "matvec", ["x"], w, W=rng.integers(-2, 2, (w, w)), b=rng.integers(-3, 3, w), shift=1),
             Node("cat", "concat", ["fc0", "x"], 2 * w)]
    W = np.concatenate([rng.integers(-2, 2, (w, w)), np.zeros((w, w), int)])
    for i in range(w): W[w + i, i] = 2
    nodes.append(Node("fc1", "matvec", ["cat"], w, W=W, b=rng.integers(-3, 3, w), shift=1))
    return Graph(nodes, "fc1", ab=ab, wb=wb, profile="v2.1")


@pytest.mark.parametrize("lanes", [1, 2])
def test_residual_and_two_bias_rows(lanes):
    rng = np.random.default_rng(6)
    for g, n in ((residual(rng), 5), (mk_graph(rng, [6, 5, 4], [1, 2], bias_rows=2), 6)):
        comp = compile_vec(g, rows=8, lanes=lanes); m = VecMachine(comp.spec)
        for _ in range(2): check(comp, m, rng.integers(0, 8, n), g)


@pytest.mark.parametrize("lanes", [1, 2])
def test_k_tiling_wide_neuron(lanes):
    """K = 20 inputs do not fit one tile of 7 data rows: a neuron sums columns of several tiles (fan > 1) inside one adder tree."""
    rng = np.random.default_rng(7)
    g = mk_graph(rng, [20, 6, 3], [2, 1])
    comp = compile_vec(g, rows=8, lanes=lanes)
    assert comp.spec.fan > 1
    m = VecMachine(comp.spec)
    for _ in range(2): check(comp, m, rng.integers(0, 8, 20), g)


@pytest.mark.parametrize("lanes", [1, 2])
def test_one_machine_many_programs(lanes):
    """The program is prompt data: two different graphs run on ONE compiled vectorized circuit."""
    from qwentensor.machine.vec import VSpec
    rows, ntile = 8, 4 * lanes
    cols = tuple(14 for _ in range(ntile)); ccls = tuple(l for _ in range(ntile) for l in (0, 1) for _ in range(14 // 2)) if lanes == 2 else None
    sp = VSpec(rows=rows, lanes=lanes, cols=cols, ccls=ccls or (), ab=3, wb=3, fan=3, smax=3)
    m = VecMachine(sp); rng = np.random.default_rng(8)
    for g, n in ((mk_graph(rng, [6, 5, 3], [1, 2]), 6), (mk_graph(rng, [5, 4, 4, 2], [2, 1, 3]), 5), (residual(rng), 5)):
        comp = compile_vec(g, rows=rows, lanes=lanes, spec_override=sp)
        for _ in range(2): check(comp, m, rng.integers(0, 8, n), g)


def test_block_align_places_each_node_on_a_fresh_block():
    rng = np.random.default_rng(9)
    g = mk_graph(rng, [6, 5, 3], [1, 2])
    plain, aligned = compile_vec(g, rows=8, lanes=2), compile_vec(g, rows=8, lanes=2, block_align=True)
    assert aligned.spec.nblk > plain.spec.nblk and aligned.spec.fan <= plain.spec.fan
    m = VecMachine(aligned.spec)
    for _ in range(2): check(aligned, m, rng.integers(0, 8, 6), g)
