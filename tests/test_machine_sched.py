"""Stage 4: Graph -> machine, end to end in the CPU circuit-IR simulator, bit-exact against reference.run."""
import numpy as np
import pytest
import torch.nn as nn

from qwentensor import frontend, quant, reference as R
from qwentensor.ir import Graph, Node
from qwentensor.machine.run import IRMachine
from qwentensor.machine.sched import compile_graph


def mk_graph(rng, dims, shifts, ab=3, wb=3, bias_rows=1, residual=False):
    nodes = [Node("x", "input", [], dims[0])]
    prev, plen = "x", dims[0]
    for i, (n, sh) in enumerate(zip(dims[1:], shifts)):
        K = plen
        W = rng.integers(-(1 << (wb - 1)), (1 << (wb - 1)), (K, n)); b = rng.integers(-(1 << (wb - 1)), (1 << (wb - 1)), n)
        nodes.append(Node(f"fc{i}", "matvec", [prev], n, W=W, b=b, shift=sh)); prev, plen = f"fc{i}", n
    return Graph(nodes, prev, ab=ab, wb=wb, profile="v2.1", bias_rows=bias_rows)


def run_machine(comp, m, xq):
    prog = comp.program(xq); p = prog.prompt(); em = prog.expected()
    pred = m.run(p, em)
    assert pred == em + [comp.spec.HALT]
    return comp.decode(em), len(p), len(em)


def test_mlp_graph_bit_exact():
    rng = np.random.default_rng(4)
    g = mk_graph(rng, [6, 5, 3], [1, 2])
    comp = compile_graph(g, rows=8)
    print(comp.spec, "ops", len(comp.ops))
    m = IRMachine(comp.spec); print(m.stats())
    for _ in range(3):
        xq = rng.integers(0, 8, 6)
        y, np_, ne = run_machine(comp, m, xq)
        assert (y == R.run(g, xq[None])[0]).all(), (y, R.run(g, xq[None]))


def test_concat_gather_groups_bias_rows():
    """Features the IR has: concat (residual), gather with duplicates (weights summed -> split over columns when > 4 bits),
    groups (conv positions), bias_rows=2 (bias beyond one weight)."""
    rng = np.random.default_rng(8)
    ab = wb = 3
    nodes = [Node("x", "input", [], 6)]
    nodes.append(Node("fc0", "matvec", ["x"], 4, W=rng.integers(-4, 4, (6, 4)), b=rng.integers(-4, 4, 4), shift=1))
    nodes.append(Node("cat", "concat", ["fc0", "x"], 10))
    nodes.append(Node("gat", "gather", ["cat"], 8, index=np.array([0, 1, 1, 1, 4, -1, 9, 9])))     # duplicates -> summed weights
    nodes.append(Node("fc1", "matvec", ["gat"], 6, W=rng.integers(-4, 4, (4, 3)), b=np.array([-8, 5, 0]), shift=0, groups=2))
    g = Graph(nodes, "fc1", ab=ab, wb=wb, profile="v2.1", bias_rows=2).validate(machine=True)
    comp = compile_graph(g, rows=8)
    print(comp.spec, len(comp.ops))
    m = IRMachine(comp.spec)
    for _ in range(2):
        xq = rng.integers(0, 8, 6)
        y, _, _ = run_machine(comp, m, xq)
        assert (y == R.run(g, xq[None])[0]).all(), (y, R.run(g, xq[None]))
