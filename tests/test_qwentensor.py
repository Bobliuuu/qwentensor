import operator
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwentensor import frontend, quant, reference as R
from qwentensor.frontend import UnsupportedOpError, from_torch
from qwentensor.ir import Graph, Node
from qwentensor.quant import QConfig

torch.manual_seed(0)
rs = np.random.default_rng(0)


class Res(nn.Module):
    def __init__(s, w=12):
        super().__init__(); s.i = nn.Linear(10, w); s.a = nn.Linear(w, w); s.b = nn.Linear(w, w); s.o = nn.Linear(w, 3)

    def forward(s, x):
        h = torch.relu(s.i(x)); h = torch.relu(h + s.b(torch.relu(s.a(h)))); return s.o(h)


class CNN(nn.Module):
    def __init__(s):
        super().__init__(); s.c1 = nn.Conv2d(1, 3, 3, padding=1); s.c2 = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        s.p = nn.AvgPool2d(2); s.f = nn.Linear(4 * 2 * 2, 5)

    def forward(s, x):
        x = torch.relu(s.c1(x)); x = torch.relu(s.c2(x)); x = s.p(x); return s.f(x.view(x.size(0), -1))


class PoolConv(nn.Module):
    def __init__(s):
        super().__init__(); s.c1 = nn.Conv2d(2, 3, 3); s.p = nn.AvgPool2d(2); s.c2 = nn.Conv2d(3, 2, 1); s.f = nn.Linear(2, 4)

    def forward(s, x):
        x = torch.relu(s.c1(x)); x = torch.relu(s.p(x)); x = torch.relu(s.c2(x)); return s.f(torch.flatten(x, 1))


class Emb(nn.Module):
    def __init__(s):
        super().__init__(); s.e = nn.Embedding(7, 4); s.l = nn.Linear(12, 5); s.o = nn.Linear(5, 3)

    def forward(s, t):
        return s.o(torch.relu(s.l(torch.flatten(s.e(t), 1))))


def _x(shape, n=64, signed=False):
    x = torch.rand(n, *shape)
    return x * 2 - 1 if signed else x


# ---------------------------------------------------------------- frontend: float IR == torch
@pytest.mark.parametrize("mk,shape,posneg", [(Res, (10,), False), (Res, (10,), True), (CNN, (1, 8, 8), False),
                                             (CNN, (1, 8, 8), True), (PoolConv, (2, 4, 4), False)])
def test_float_graph_matches_torch(mk, shape, posneg):
    m = mk().double(); g = from_torch(m, shape, input_posneg=posneg)
    x = _x(shape, signed=posneg).double()
    ref = m(x).detach().numpy()
    out = R.run_float(g, R.encode_input(g, x.numpy()))
    assert np.allclose(out, ref, atol=1e-9)


def test_embedding_float_graph_matches_torch():
    m = Emb().double(); g = from_torch(m, (3,), input_kind="index")
    t = torch.randint(0, 7, (20, 3))
    assert np.allclose(R.run_float(g, R.encode_input(g, t.numpy())), m(t).detach().numpy(), atol=1e-9)
    assert g.input.encode == "onehot" and g.input.out_len == 21


def test_fusion_counts():
    g = from_torch(Res(), (10,))
    assert [n.op for n in g.nodes] == ["input", "matvec", "matvec", "concat", "matvec", "matvec"]   # residual add costs no node
    lin2 = nn.Sequential(nn.Linear(6, 5), nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2))
    assert len(from_torch(lin2, (6,)).matvecs()) == 2                                               # Linear->Linear merged


# ---------------------------------------------------------------- unsupported ops are rejected with a clear error
class _Mod(nn.Module):
    def __init__(s, f): super().__init__(); s.l = nn.Linear(6, 6); s.f = f

    def forward(s, x): return s.f(s.l(x))


class _Fn(nn.Module):
    def __init__(s, fn): super().__init__(); s.l = nn.Linear(6, 6); s.fn = fn

    def forward(s, x): return s.fn(s.l(x))


@pytest.mark.parametrize("mod,word", [(nn.Softmax(-1), "Softmax"), (nn.LayerNorm(6), "LayerNorm"), (nn.GELU(), "GELU"),
                                      (nn.Sigmoid(), "Sigmoid"), (nn.Tanh(), "Tanh"), (nn.SiLU(), "SiLU"),
                                      (nn.BatchNorm1d(6), "BatchNorm1d")])
def test_rejects_unsupported_modules(mod, word):
    with pytest.raises(UnsupportedOpError) as e:
        from_torch(_Mod(mod), (6,))
    assert word in str(e.value) and "unsupported op" in str(e.value)


@pytest.mark.parametrize("fn,word", [(lambda x: F.softmax(x, -1), "softmax"), (torch.sigmoid, "sigmoid"), (F.gelu, "gelu"),
                                     (lambda x: x * x, "mul"), (lambda x: torch.matmul(x, x), "matmul"), (lambda x: F.layer_norm(x, (6,)), "layer_norm")])
def test_rejects_unsupported_functions(fn, word):
    with pytest.raises(UnsupportedOpError) as e:
        from_torch(_Fn(fn), (6,))
    assert word in str(e.value)


def test_rejects_maxpool_and_attention_and_unactivated_conv():
    class MP(nn.Module):
        def __init__(s): super().__init__(); s.c = nn.Conv2d(1, 2, 3); s.p = nn.MaxPool2d(2)
        def forward(s, x): return s.p(torch.relu(s.c(x)))
    with pytest.raises(UnsupportedOpError, match="MaxPool2d"): from_torch(MP(), (1, 6, 6))

    class At(nn.Module):
        def __init__(s): super().__init__(); s.a = nn.MultiheadAttention(4, 1)
        def forward(s, x): return s.a(x, x, x)[0]
    with pytest.raises(UnsupportedOpError): from_torch(At(), (3, 4))

    class CC(nn.Module):                       # conv fed by an un-activated (possibly negative) tensor
        def __init__(s): super().__init__(); s.c = nn.Conv2d(1, 2, 3); s.d = nn.Conv2d(2, 2, 3)
        def forward(s, x): return s.d(s.c(x))
    with pytest.raises(UnsupportedOpError, match="un-activated"): from_torch(CC(), (1, 8, 8))


def test_message_names_the_node():
    with pytest.raises(UnsupportedOpError) as e:
        from_torch(_Mod(nn.Softmax(-1)), (6,))
    assert "fx node 'f'" in str(e.value) and "argmax" in str(e.value)


# ---------------------------------------------------------------- IR validation
def _tiny(W, b, shift=0, profile="v2.1", act="relu_clamp"):
    return Graph([Node("x", "input", [], W.shape[0], scale=np.ones(W.shape[0])),
                  Node("m", "matvec", ["x"], W.shape[1], W=W, b=b, shift=shift, act=act)], "m", profile=profile)


def test_validate_ranges():
    _tiny(np.full((3, 2), 15), np.zeros(2, int)).validate()
    with pytest.raises(AssertionError, match="weight out of"): _tiny(np.full((3, 2), 16), np.zeros(2, int)).validate()
    with pytest.raises(AssertionError, match="bias out of"): _tiny(np.ones((3, 2), int), np.full(2, 16)).validate()
    with pytest.raises(AssertionError, match="shift"): _tiny(np.ones((3, 2), int), np.zeros(2, int), shift=2, profile="v2.0").validate()
    with pytest.raises(AssertionError, match="not machine-native"): _tiny(np.ones((3, 2), int), np.zeros(2, int), act="linear").validate(machine=True)


# ---------------------------------------------------------------- reference vs an independent numpy computation
def test_reference_floor_shift_semantics():
    W = np.array([[3, -5], [-7, 2], [1, 1]]); b = np.array([-3, 4])
    g = _tiny(W, b, shift=2).validate()
    x = rs.integers(0, 32, (200, 3))
    out = R.run(g, x)
    for i in range(len(x)):                                           # python-int loop, floor division
        for n in range(2):
            acc = sum(int(x[i, k]) * int(W[k, n]) for k in range(3)) + int(b[n])
            assert out[i, n] == min(max(acc // 4, 0), 31)


def _mlp_graph_indep(Xq, g):
    """Independent MLP execution straight from the graph's integer tensors, python ints, no reference.py."""
    ms = g.matvecs(); out = []
    for row in Xq:
        v = [int(a) for a in row]
        for n in ms:
            K, N = n.W.shape
            nv = []
            for j in range(N):
                acc = sum(v[k] * int(n.W[k, j]) for k in range(K)) + int(n.b[j])
                acc >>= n.shift
                nv.append(min(max(acc, 0), 31) if n.act == "relu_clamp" else acc)
            v = nv
        out.append(v)
    return np.array(out)


def test_quantized_mlp_matches_independent_numpy():
    m = nn.Sequential(nn.Linear(9, 14), nn.ReLU(), nn.Linear(14, 11), nn.ReLU(), nn.Linear(11, 4)).double()
    X = rs.random((300, 9))
    for prof in ("v2.0", "v2.1"):
        g = quant.quantize(from_torch(m, (9,)), X, QConfig(prof, bias_rows=2))
        Xq = R.encode_input(g, X[:40])
        assert np.array_equal(R.run(g, Xq), _mlp_graph_indep(Xq, g))
        assert all(n.shift == 0 for n in g.matvecs()) if prof == "v2.0" else True


def test_quantized_cnn_matches_independent_conv_loops():
    """Conv + pool-folded Linear against nested-loop numpy in the LOGICAL (C,H,W) layout, using only the graph's integer weights."""
    m = CNN().double(); X = rs.random((200, 64))
    g = quant.quantize(from_torch(m, (1, 8, 8)), X, QConfig("v2.1", bias_rows=4))
    xq = R.encode_input(g, X[:12]); got = R.run(g, xq)
    mv = g.matvecs(); c1, c2, fc = mv
    assert c1.groups == 64 and c2.groups == 16 and fc.groups == 1
    def conv(img, node, stride, pad, Cin, Co, k=3):
        Wk = node.W.T.reshape(Co, Cin, k, k); H = img.shape[1]; Ho = (H + 2 * pad - k) // stride + 1
        padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad))); out = np.zeros((Co, Ho, Ho), dtype=np.int64)
        for co in range(Co):
            for i in range(Ho):
                for j in range(Ho):
                    acc = int((padded[:, i * stride:i * stride + k, j * stride:j * stride + k] * Wk[co]).sum()) + int(node.b[co])
                    out[co, i, j] = min(max(acc >> node.shift, 0), 31)
        return out
    for s in range(len(xq)):
        a1 = conv(xq[s].reshape(1, 8, 8), c1, 1, 1, 1, 3); a2 = conv(a1, c2, 2, 1, 3, 4)       # (4,4,4) logical
        # grouped-matvec outputs are position-major (p*Co + co); the Flatten is free, so fc's rows follow that physical order
        acc = a2.transpose(1, 2, 0).reshape(-1) @ fc.W + fc.b
        assert fc.W.shape[0] == 4 * 4 * 4
        # the 2x2 AvgPool is folded into fc (4 equal rows of weight w/4 per pooled element), so fc sees the un-pooled 4x4 map
        y = np.clip(acc >> fc.shift, 0, 31) if fc.act == "relu_clamp" else acc >> fc.shift
        assert np.array_equal(y, got[s]), s


def test_quantized_close_to_float():
    """Dequantized integer output tracks the float output at 5 bits (v2.1), on uniform random inputs and a seeded random-weight network.
    Measured (seed 0): residual MLP about 0.97, CNN about 0.89. The CNN bar is 0.85 because a random-weight CNN on random inputs
    quantizes worse than the trained nets in examples/ (those are scored by accuracy there). This checks the quantizer, not the machine."""
    for mk, shape, bar in [(Res, (10,), 0.95), (CNN, (1, 8, 8), 0.85)]:
        torch.manual_seed(0); m = mk().double(); X = np.random.default_rng(0).random((300, int(np.prod(shape))))    # own RNG: independent of test order
        fg = from_torch(m, shape); g = quant.quantize(fg, X, QConfig("v2.1", bias_rows=4, logits="linear"))
        yf = R.run_float(fg, R.encode_input(fg, X[:100])); yq = R.run(g, R.encode_input(g, X[:100])) * g.node(g.output).scale
        corr = np.corrcoef(yf.ravel(), yq.ravel())[0, 1]; print(mk.__name__, 'corr', round(float(corr), 4))
        assert corr > bar, (mk.__name__, corr)


def test_posneg_and_onehot_quantize_run():
    m = Res().double(); X = rs.standard_normal((200, 10))
    g = quant.quantize(from_torch(m, (10,), input_posneg=True), X, QConfig("v2.1", bias_rows=4)).validate(machine=True)
    assert g.input.out_len == 20 and R.run(g, R.encode_input(g, X[:5])).shape == (5, 3)
    with pytest.raises(ValueError, match="negative"): quant.quantize(from_torch(m, (10,)), X)
    e = Emb().double(); T = rs.integers(0, 7, (100, 3))
    ge = quant.quantize(from_torch(e, (3,), input_kind="index"), T, QConfig("v2.1", bias_rows=4))
    assert R.run(ge, R.encode_input(ge, T[:4])).shape == (4, 3)


def test_serialization_roundtrip(tmp_path):
    m = CNN().double(); X = rs.random((100, 64))
    g = quant.quantize(from_torch(m, (1, 8, 8)), X, QConfig("v2.1"))
    g.save(tmp_path / "g.json"); g2 = Graph.load(tmp_path / "g.json").validate()
    assert np.array_equal(R.run(g, R.encode_input(g, X[:9])), R.run(g2, R.encode_input(g2, X[:9])))
    assert g.cost()["total_tile_matvecs"] == g2.cost()["total_tile_matvecs"]


def test_machine_validity_and_qat_not_worse():
    m = nn.Sequential(nn.Linear(15, 15), nn.ReLU(), nn.Linear(15, 10)).double()
    X = rs.random((400, 15)); y = (X[:, :10] .argmax(1)).astype(np.int64)
    g = quant.quantize(from_torch(m, (15,)), X, QConfig("v2.0", input_max=12)).validate(machine=True)
    g2 = quant.qat(g, X, y, epochs=6).validate(machine=True)
    assert quant.accuracy(g2, X, y) >= quant.accuracy(g, X, y)
    assert all(n.shift == 0 for n in g2.matvecs())


def test_digits_mlp_accuracy():
    pytest.importorskip("sklearn")
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from common import digits, train_float, torch_acc
    Xtr, Xte, ytr, yte = digits("pool15")
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(15, 15), nn.ReLU(), nn.Linear(15, 10)); train_float(m, Xtr, ytr, epochs=100)
    fg = from_torch(m, (15,)); facc = quant.float_accuracy(fg, Xte, yte)
    assert abs(facc - torch_acc(m, Xte, yte)) < 1e-9 and facc > 0.85
    g = quant.quantize(fg, Xtr, QConfig("v2.1", bias_rows=4))
    assert quant.accuracy(g, Xte, yte) > facc - 0.10
