"""Exact skipping on the chip path: a real transformers Qwen3.5 model holding the machine circuit (qwentpu.compiler.tiny), CPU.
The skipping path must produce the same layer outputs, the same emitted tokens and the same recurrent states as the stock path."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
tiny = pytest.importorskip("qwentpu.compiler.tiny")

from qwentensor.ir import Graph, Node
from qwentensor.machine.circuit import build
from qwentensor.machine.gpu import install
from qwentensor.machine.sched import compile_graph
from qwentensor.machine import skip as S


def small_graph():
    rng = np.random.default_rng(3)
    nodes = [Node("x", "input", [], 8)]
    nodes.append(Node("fc0", "matvec", ["x"], 5, W=rng.integers(-3, 4, (8, 5)), b=rng.integers(-8, 8, 5), shift=2))
    nodes.append(Node("fc1", "matvec", ["fc0"], 4, W=rng.integers(-3, 4, (5, 4)), b=rng.integers(-8, 8, 4), shift=1))
    return Graph(nodes, "fc1", ab=5, wb=5, profile="v2.1", bias_rows=1), rng


@pytest.fixture(scope="module")
def machine():
    g_, rng = small_graph()
    comp = compile_graph(g_, rows=16); sp = comp.spec
    install(sp)
    g, d = build(sp)
    model = tiny.build(g, list(range(sp.nvocab)), [next(iter(e.t)) for e in d["code"]])
    return comp, sp, model, rng


def dense_hidden(model, ids, past, nl):
    m = model.model; h = m.embed_tokens(ids)
    for layer in m.layers[:nl]:
        h = layer(h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=past)
    return m.norm(h)


def test_compact_layers_find_dead_structure(machine):
    comp, sp, model, _ = machine
    nl = model.config.num_hidden_layers - 1
    path = S.SkipChip(model, 0, readout="lm_head", first=0, last=nl)
    assert all(not c.dense for c in path.cl.values())          # the circuit has no conv history taps
    assert sum(c.nh for c in path.cl.values()) < nl * 16        # some heads are structurally dead
    assert sum(c.nu for c in path.cl.values()) < nl * model.config.intermediate_size // 4


def test_skip_path_is_bit_identical(machine):
    from transformers import DynamicCache
    comp, sp, model, rng = machine
    nl = model.config.num_hidden_layers - 1
    xq = rng.integers(0, 32, 8); prog = comp.program(xq); p = prog.prompt(); want = prog.expected()
    skip = S.SkipChip(model, 0, readout="lm_head", first=0, last=nl)
    head = model.lm_head.weight[: sp.nvocab]
    pd, ps = DynamicCache(config=model.config), DynamicCache(config=model.config)
    n = min(len(want), 12)
    with torch.no_grad():
        ids = torch.tensor([p])                                    # the prompt (chunk) goes through the stock layers on both sides
        td = (dense_hidden(model, ids, pd, nl) @ head.T).argmax(-1); ts = skip.tokens(ids, ps)
        assert torch.equal(td, ts)
        nxt = int(ts[0, -1]); em = []
        for _ in range(n):
            em.append(nxt); ids = torch.tensor([[nxt]])
            h_dense = dense_hidden(model, ids, pd, nl); h_skip = skip.hidden(ids, ps)
            assert torch.equal(h_dense, h_skip)                    # every layer's output, bit for bit
            nxt = int((h_skip[..., skip.cols] @ skip.head_c.T).argmax(-1)[0, -1])
            assert nxt == int((h_dense @ head.T).argmax(-1)[0, -1])
        for t in (sp.NOP, sp.HALT, sp.NOP, sp.NOP):                # tokens the program never emits: teacher-forced, still bit-equal
            ids = torch.tensor([[t]])
            assert torch.equal(dense_hidden(model, ids, pd, nl), skip.hidden(ids, ps))
    assert em == want[:n]
    for L in range(nl):
        assert torch.equal(pd.layers[L].recurrent_states[0], ps.layers[L].recurrent_states[0])
    assert skip.stats["attn_skipped"] > 0 and skip.stats["attn_run"] > 0
