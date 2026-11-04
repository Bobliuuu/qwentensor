"""One general machine, several graphs from the frontend-style IR, u5s5, CPU simulator vs reference.run (slow)."""
import numpy as np
import pytest

from qwentensor import reference as R
from qwentensor.ir import Graph, Node
from qwentensor.machine.run import IRMachine
from qwentensor.machine.sched import compile_graph
from qwentensor.machine.spec import Spec
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "qwentensor", "machine", "data")


def residual(rng, w=7):
    nodes = [Node("x", "input", [], w), Node("fc0", "matvec", ["x"], w, W=rng.integers(-3, 4, (w, w)), b=rng.integers(-8, 8, w), shift=3),
             Node("cat", "concat", ["fc0", "x"], 2 * w)]
    W = np.concatenate([rng.integers(-3, 4, (w, w)), np.zeros((w, w), int)])
    for i in range(w): W[w + i, i] = 8
    nodes.append(Node("fc1", "matvec", ["cat"], w, W=W, b=rng.integers(-8, 8, w), shift=3))
    return Graph(nodes, "fc1", ab=5, wb=5, profile="v2.1")


def test_digits_and_residual_one_machine():
    sp = Spec(rows=16, cols=(15, 12, 8), ab=5, wb=5, nops=40, fan=2, smax=7)
    m = IRMachine(sp); rng = np.random.default_rng(2)
    gd = Graph.load(os.path.join(DATA, "digits_v21.json")); z = np.load(os.path.join(DATA, "digits_samples_v21.npz"))
    for g, xq in [(gd, z["xq"][0]), (residual(rng), rng.integers(0, 32, 7))]:
        comp = compile_graph(g, rows=16, spec_override=sp)
        prog = comp.program(xq); p = prog.prompt(); em = prog.expected()
        assert m.run(p, em) == em + [0]
        assert (comp.decode(em) == R.run(g, xq[None])[0]).all()
