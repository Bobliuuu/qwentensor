"""Paged execution (weight streaming, docs/STREAMING.md): networks larger than one machine run as pages on one shared Spec,
bit-exact against reference.run, in the CPU circuit-IR simulator (every bf16 / silu / state-row exactness condition asserted)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

import qwentensor as qt
from qwentensor import reference as R
from qwentensor.ir import Graph, Node
from qwentensor.machine.pages import compile_paged, golden_executor, sim_executor, PageMismatch
from qwentensor.machine.sched import compile_graph, DoesNotFit


def mk_mlp(rng, dims, shifts, ab=3, wb=3, bias_rows=1):
    lo, hi = -(1 << (wb - 1)), (1 << (wb - 1))
    nodes = [Node("x", "input", [], dims[0])]; prev, plen = "x", dims[0]
    for i, (n, sh) in enumerate(zip(dims[1:], shifts)):
        W = rng.integers(lo, hi, (plen, n)); b = rng.integers(lo, hi, n)
        nodes.append(Node(f"fc{i}", "matvec", [prev], n, W=W, b=b, shift=sh)); prev, plen = f"fc{i}", n
    return Graph(nodes, prev, ab=ab, wb=wb, profile="v2.1", bias_rows=bias_rows)


def mk_res(rng, d_in, w, shifts, ab=5, wb=5):
    """in -> w -> res block (identity rows over concat) -> 10 : the skip tensor must survive several pages."""
    sc = 1 << shifts[2]
    nodes = [Node("x", "input", [], d_in),
             Node("fc0", "matvec", ["x"], w, W=rng.integers(-3, 4, (d_in, w)), b=rng.integers(-8, 8, w), shift=shifts[0]),
             Node("fc1", "matvec", ["fc0"], w, W=rng.integers(-3, 4, (w, w)), b=rng.integers(-8, 8, w), shift=shifts[1]),
             Node("cat", "concat", ["fc1", "fc0"], 2 * w)]
    W = np.concatenate([rng.integers(-3, 4, (w, w)), np.zeros((w, w), int)])
    for i in range(w): W[w + i, i] = min(sc, 15)                       # residual identity rows (scaled by 2^shift, clipped to the weight range)
    nodes.append(Node("fc2", "matvec", ["cat"], w, W=W, b=rng.integers(-8, 8, w), shift=shifts[2]))
    nodes.append(Node("out", "matvec", ["fc2"], 10, W=rng.integers(-3, 4, (w, 10)), b=rng.integers(-8, 8, 10), shift=shifts[3]))
    return Graph(nodes, "out", ab=ab, wb=wb, profile="v2.1")


def run_paged(pc, g, xq, execute):
    got = np.stack([pc.run(x, execute)[0] for x in xq]); ref = R.run(g, xq)
    assert (got == ref).all(), (got[(got != ref).any(1)][:1], ref[(got != ref).any(1)][:1])


def test_pages_respect_budget_and_cover_every_neuron():
    rng = np.random.default_rng(0); g = mk_mlp(rng, [12, 14, 10, 6], [2, 1, 2])
    pc = compile_paged(g, rows=8, max_cols=10, max_ops=8)
    assert len(pc.pages) > 1
    seen = [k for p in pc.pages for k in p.dst]
    assert len(seen) == len(set(seen)) == 30
    assert all(len(p.neurons) <= 8 and p.comp.spec == pc.spec for p in pc.pages)


def test_small_mlp_in_pages_matches_reference_in_circuit():
    rng = np.random.default_rng(0); g = mk_mlp(rng, [12, 14, 10, 6], [2, 1, 2])
    pc = compile_paged(g, rows=8, max_cols=10, max_ops=8)
    run_paged(pc, g, rng.integers(0, 8, (3, 12)), sim_executor(pc.spec))


def test_skip_connection_across_pages_in_circuit():
    rng = np.random.default_rng(3); g = mk_res(rng, 20, 8, [3, 2, 3, 2])
    pc = compile_paged(g, rows=16, max_cols=12, max_ops=6)
    assert len(pc.pages) >= 4
    run_paged(pc, g, rng.integers(0, 32, (2, 20)), sim_executor(pc.spec))


def test_gather_and_groups_in_pages():
    rng = np.random.default_rng(5)
    nodes = [Node("x", "input", [], 12),
             Node("g", "gather", ["x"], 8, index=np.array([0, 1, 2, 3, 2, 3, 4, 5])),                          # two overlapping 4-wide windows
             Node("c", "matvec", ["g"], 12, W=rng.integers(-4, 4, (4, 6)), b=rng.integers(-4, 4, 6), shift=1, groups=2)]
    g = Graph(nodes, "c", ab=3, wb=3, profile="v2.1").validate(machine=True)
    pc = compile_paged(g, rows=8, max_cols=6, max_ops=4)
    assert len(pc.pages) >= 3
    run_paged(pc, g, rng.integers(0, 8, (3, 12)), sim_executor(pc.spec))


def test_deep_mlp_beyond_one_machine():
    """142 neurons (> 128 op slots), about 400 columns: DoesNotFit on one machine, exact in pages."""
    rng = np.random.default_rng(1); g = mk_mlp(rng, [44, 44, 44, 44, 10], [5, 5, 5, 5], ab=5, wb=5)
    for n in g.matvecs(): n.W = np.clip(n.W, -3, 3)
    with pytest.raises(DoesNotFit): compile_graph(g)
    pc = compile_paged(g, rows=16, max_cols=45, max_ops=60)
    assert pc.spec.ncols <= 126 and len(pc.pages) >= 8 and sum(len(p.neurons) for p in pc.pages) == 142
    xq = rng.integers(0, 32, (8, 44))
    run_paged(pc, g, xq, golden_executor())                                       # token golden over many inputs
    run_paged(pc, g, xq[:1], sim_executor(pc.spec))                                # the circuit, one input, every page


def test_residual_mlp_beyond_one_machine():
    rng = np.random.default_rng(2); g = mk_res(rng, 44, 20, [5, 4, 5, 5])
    pc = compile_paged(g, rows=16, max_cols=45, max_ops=60)
    assert len(pc.pages) >= 4
    xq = rng.integers(0, 32, (8, 44))
    run_paged(pc, g, xq, golden_executor())
    run_paged(pc, g, xq[:1], sim_executor(pc.spec))


def test_wide_neurons_still_do_not_fit():
    """64 inputs at shift 6 need fan 5: the requantizer's carry staircase does not fit its window (docs/STREAMING.md 3.2)."""
    rng = np.random.default_rng(4); g = mk_mlp(rng, [64, 12, 10], [6, 6], ab=5, wb=5)
    with pytest.raises(DoesNotFit, match="fan"): compile_paged(g, rows=16, max_cols=60, max_ops=60)


def test_single_neuron_over_the_column_budget():
    rng = np.random.default_rng(6); g = mk_mlp(rng, [30, 4], [2], ab=3, wb=3)
    with pytest.raises(DoesNotFit, match="single neuron"): compile_paged(g, rows=8, max_cols=2, max_ops=4)


def test_circuit_disagreement_is_reported():
    rng = np.random.default_rng(0); g = mk_mlp(rng, [12, 14, 10, 6], [2, 1, 2])
    pc = compile_paged(g, rows=8, max_cols=10, max_ops=8)
    pg = pc.pages[0]; prog = pg.comp.program(rng.integers(0, 8, len(pg.src)))

    class Corrupt:
        prompt = staticmethod(prog.prompt)
        @staticmethod
        def expected():
            em = list(prog.expected()); em[1] = (em[1] + 1) % pc.spec.nvocab; return em
    with pytest.raises(PageMismatch): sim_executor(pc.spec)(Corrupt)


# ------------------------------------------------------------------------------------------------------------ public API
def test_api_paged_torch_model_bit_exact_and_roundtrip(tmp_path):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    X = rng.random((200, 20)); y = rng.integers(0, 4, 200)
    model = nn.Sequential(nn.Linear(20, 18), nn.ReLU(), nn.Linear(18, 14), nn.ReLU(), nn.Linear(14, 4))
    prog = qt.compile(model, X, input_shape=(20,), paged=True, max_cols=12, max_ops=8, rows=8)
    assert isinstance(prog, qt.PagedProgram) and len(prog.paged.pages) > 1
    out = prog.run(X[:2])                                       # raises MachineMismatch unless == reference.run
    assert out.shape == (2, 4) and prog.last["pages"] == len(prog.paged.pages)
    assert "pages" in prog.explain() and prog.cost()["pages"] == len(prog.paged.pages)
    prog.save(tmp_path / "p.json"); again = qt.load(str(tmp_path / "p.json"))
    assert isinstance(again, qt.PagedProgram) and (again.reference(X[:2]) == prog.reference(X[:2])).all()
    with pytest.raises(TypeError): qt.compile(model, X, input_shape=(20,), paged=True, fan=3)
