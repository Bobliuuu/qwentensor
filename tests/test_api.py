"""qt.compile / Program: user path on the CPU simulator (small networks so the circuit simulation stays quick)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

import qwentensor as qt


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0); torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    X = rng.random((200, 6)); y = (X[:, :3].sum(1) > X[:, 3:].sum(1)).astype(int) + (X[:, 0] > 0.5)
    m = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3)).double()
    opt = torch.optim.Adam(m.parameters(), 0.05); Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(150):
        loss = nn.functional.cross_entropy(m(Xt), yt); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval(), X, y


def test_compile_run_matches_reference(tiny):
    m, X, y = tiny
    prog = qt.compile(m, X, input_shape=(6,), backend="sim")
    out = prog.run(X[:3])
    assert out.shape == (3, 3) and (out == prog.reference(X[:3])).all()
    assert prog.check(X[3:5])
    assert prog.run(X[0]).shape == (1, 3)                       # single sample
    assert prog.predict(X[:3]).shape == (3,)
    assert 0 <= prog.accuracy(X[:3], y[:3]) <= 1
    assert prog.last["prompt_tokens"] > 0


def test_explain_cost_save_load(tiny, tmp_path):
    m, X, y = tiny
    prog = qt.compile(m, X, input_shape=(6,), labels=y, qat=dict(epochs=2))
    c = prog.cost()
    assert c["neurons"] == 8 and c["ops"] == 8 and c["total_tile_matvecs"] >= 2
    assert "qwentensor Program" in prog.explain() and "8 neurons" in prog.explain()
    p = tmp_path / "p.json"; prog.save(p)
    p2 = qt.load(p)
    assert (p2.run(X[:2]) == prog.run(X[:2])).all()


def test_float_int_graph_inputs(tiny):
    m, X, y = tiny
    from qwentensor import frontend
    fg = frontend.from_torch(m, (6,))
    p1 = qt.compile(fg, X)
    p2 = qt.compile(p1.graph, X)                                # already-integer graph
    assert (p1.reference(X[:5]) == p2.reference(X[:5])).all()
    assert p1.accuracy(X, y, on="float") > 0.5


def test_does_not_fit_message():
    m = nn.Sequential(nn.Linear(16, 100), nn.ReLU(), nn.Linear(100, 60), nn.ReLU(), nn.Linear(60, 10))
    with pytest.raises(qt.DoesNotFit) as e:
        qt.compile(m, np.random.default_rng(0).random((50, 16)), input_shape=(16,))
    msg = str(e.value)
    assert "170 neurons" in msg and "128" in msg and "To fit" in msg and "mv0: 100 neurons" in msg


def test_unsupported_op_clear():
    m = nn.Sequential(nn.Linear(4, 4), nn.Softmax(dim=1), nn.Linear(4, 3))
    with pytest.raises(qt.UnsupportedOpError) as e:
        qt.compile(m, np.random.default_rng(0).random((10, 4)), input_shape=(4,))
    assert "softmax" in str(e.value).lower()


def test_argument_errors(tiny):
    m, X, _ = tiny
    with pytest.raises(ValueError, match="input_shape"): qt.compile(m, X)
    with pytest.raises(ValueError, match="calib"): qt.compile(m, None, input_shape=(6,))
    with pytest.raises(ValueError, match="labels"): qt.compile(m, X, input_shape=(6,), qat=True)
    with pytest.raises(TypeError): qt.compile("not a model", X, input_shape=(6,))
    with pytest.raises(ValueError, match="backend"): qt.compile(m, X, input_shape=(6,), backend="tpu")


def test_qwen_backend_needs_gpu(tiny, monkeypatch):
    m, X, _ = tiny
    monkeypatch.delenv("QWEN_DIR", raising=False)
    prog = qt.compile(m, X, input_shape=(6,), backend="qwen")
    with pytest.raises(RuntimeError, match="QWEN_DIR"): prog.run(X[:1])
