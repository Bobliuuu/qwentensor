"""Vectorized machine on the real merged Qwen3.5 (A100).  Host = argmax; append.
usage: python scripts/gpu_vec.py exact|bench [variants]      variants: scalar vec1 vec2 (default: scalar vec1 vec2)
env: QWEN_DIR = checkpoint dir, QWENTPU_SRC = qwenTPU repo, TAG = label printed in the log, N = number of digits (default 20)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), *[p for p in [os.environ.get("QWENTPU_SRC")] if p]]
import numpy as np, torch
from qwentpu.merge import build_merged, instantiate
from qwentpu.driver import TpuSession
from qwentensor import reference as R
from qwentensor.ir import Graph, Node
from qwentensor.machine.gpu import install
from qwentensor.machine.run import generate
from qwentensor.machine.sched import compile_graph, DoesNotFit
from qwentensor.machine.vec import VSpec, compile_vec, install_vec

SRC = os.environ.get("QWEN_DIR") or sys.exit("set QWEN_DIR to a Qwen3.5 checkpoint directory (and CUDA)")
TAG = os.environ.get("TAG", os.path.basename(SRC.rstrip("/")))
N = int(os.environ.get("N", 20))
DATA = os.path.join(HERE, "..", "qwentensor", "machine", "data")
bad = 0


def digits():
    g = Graph.load(os.path.join(DATA, "digits_v21.json")); z = np.load(os.path.join(DATA, "digits_samples_v21.npz"))
    return g, list(z["xq"][:N]), z["y"][:N]


def make(kind, g):
    if kind == "scalar":
        comp = compile_graph(g, rows=16); return comp, comp.spec, install
    lanes = int(kind[-1]); comp = compile_vec(g, rows=16, lanes=lanes); return comp, comp.spec, install_vec


def load(kind, g):
    comp, sp, inst = make(kind, g); inst(sp); t0 = time.time()
    tc, sd, v0, lay, gg = build_merged(SRC, mute_qwen=False)
    model = instantiate(tc, sd, "cuda")
    print(f"[{kind}] model {TAG} {sp}\n[{kind}] chip layers {gg.n_layers()} hidden {lay.hidden} vocab {sp.nvocab} build {time.time()-t0:.0f}s", flush=True)
    return comp, sp, model, v0, gg


def rand_graph(rng, dims, shifts, ab=5, wb=5):
    nodes = [Node("x", "input", [], dims[0])]; prev, plen = "x", dims[0]
    for i, (n, sh) in enumerate(zip(dims[1:], shifts)):
        nodes.append(Node(f"fc{i}", "matvec", [prev], n, W=rng.integers(-3, 4, (plen, n)), b=rng.integers(-8, 8, n), shift=sh)); prev, plen = f"fc{i}", n
    return Graph(nodes, prev, ab=ab, wb=wb, profile="v2.1")


def residual_graph(rng, w=7, ab=5, wb=5):
    nodes = [Node("x", "input", [], w), Node("fc0", "matvec", ["x"], w, W=rng.integers(-3, 4, (w, w)), b=rng.integers(-8, 8, w), shift=3),
             Node("cat", "concat", ["fc0", "x"], 2 * w)]
    W = np.concatenate([rng.integers(-3, 4, (w, w)), np.zeros((w, w), int)])
    for i in range(w): W[w + i, i] = 8
    nodes.append(Node("fc1", "matvec", ["cat"], w, W=W, b=rng.integers(-8, 8, w), shift=3))
    return Graph(nodes, "fc1", ab=ab, wb=wb, profile="v2.1")


def exact(kind):
    """Digits MLP: N digits x 2 readouts, emitted tokens == numpy token golden AND outputs == reference.run."""
    global bad
    g, xs, y = digits(); comp, sp, model, v0, gg = load(kind, g)
    for ro in ("code", "lm_head"):
        ok = 0; t1 = time.time()
        for xq in xs:
            prog = comp.program(xq); p = prog.prompt(); want = prog.expected()
            em = generate(TpuSession(model, v0, fast=True, readout=ro).feed, p, sp.HALT)
            good = em == want and (comp.decode(em) == R.run(g, np.asarray(xq)[None])[0]).all(); ok += good
            if not good: print("MISMATCH", kind, ro, flush=True)
        bad += len(xs) - ok
        print(f"[{kind}] digits MLP 15-15-10, readout {ro}: {ok}/{len(xs)} exact (emitted tokens == numpy token golden AND outputs == reference.run); "
              f"prompt {len(p)} tok, emitted {len(want)}; {time.time()-t1:.1f}s", flush=True)
    ref = np.stack([R.run(g, x[None])[0] for x in xs]); print(f"[{kind}] argmax acc on these {len(xs)}:", (ref.argmax(1) == y).mean(), flush=True)
    del model; torch.cuda.empty_cache()


def general_spec(lanes):
    """3 blocks; every tile has 16 columns (one lane) or 12 + 12 columns of lane classes 0 and 1 (two lanes)."""
    per = 16 // lanes if lanes == 1 else 12
    return VSpec(rows=16, lanes=lanes, cols=tuple(per * lanes for _ in range(3 * lanes)),
                 ccls=tuple(l for _ in range(3 * lanes) for l in range(lanes) for _ in range(per)) if lanes > 1 else (), ab=5, wb=5, fan=3, smax=7)


def exact_general(kind):
    """ONE compiled vectorized machine, four different programs (only the prompt differs).  One lane only: a two-lane machine that
    holds all four programs has 28 chip layers (297 heads), so the two-lane generality is checked on the CPU simulator instead."""
    global bad
    assert kind == "vec1"
    lanes = int(kind[-1]); rng = np.random.default_rng(1)
    gd, xs, _ = digits(); sp = general_spec(lanes)
    graphs = [("digits MLP 15-15-10", gd, xs[:6]), ("random MLP 20-8-4 (K=20 spans tiles)", rand_graph(rng, [20, 8, 4], [3, 2]), None),
              ("random 4-layer 8-5-5-5-4", rand_graph(rng, [8, 5, 5, 5, 4], [2, 2, 3, 1]), None), ("residual block (concat, identity rows)", residual_graph(rng), None)]
    install_vec(sp); t0 = time.time()
    tc, sd, v0, lay, gg = build_merged(SRC, mute_qwen=False); model = instantiate(tc, sd, "cuda")
    print(f"[{kind}:general] model {TAG} {sp}\n[{kind}:general] chip layers {gg.n_layers()} hidden {lay.hidden} vocab {sp.nvocab} build {time.time()-t0:.0f}s", flush=True)
    for name, gr, x6 in graphs:
        try: comp = compile_vec(gr, rows=16, lanes=lanes, spec_override=sp)
        except DoesNotFit as e: print("SKIP", name, e); continue
        x6 = x6 if x6 is not None else [rng.integers(0, 32, gr.input.out_len) for _ in range(6)]
        ok = 0; t1 = time.time()
        for xq in x6:
            prog = comp.program(xq); p = prog.prompt(); want = prog.expected()
            em = generate(TpuSession(model, v0, fast=True, readout="code").feed, p, sp.HALT)
            ok += em == want and (comp.decode(em) == R.run(gr, np.asarray(xq)[None])[0]).all()
        bad += len(x6) - ok
        print(f"[{kind}:general] {name}: {ok}/{len(x6)} exact; prompt {len(p)} tok, emitted {len(want)}; {time.time()-t1:.1f}s", flush=True)
    del model; torch.cuda.empty_cache()


def bench(kind):
    global bad
    g, xs, _ = digits(); comp, sp, model, v0, gg = load(kind, g); sync = torch.cuda.synchronize

    def infer(xq, fast):
        prog = comp.program(xq); p = prog.prompt(); want = prog.expected()
        sess = TpuSession(model, v0, fast=fast, readout="code")
        sync(); t0 = time.perf_counter()
        nxt = sess.feed(list(p))[-1]; sync(); t1 = time.perf_counter()
        em = []
        while nxt != sp.HALT:
            em.append(nxt); nxt = sess.feed([nxt])[-1]
        sync(); t2 = time.perf_counter()
        return em == want and (comp.decode(em) == R.run(g, np.asarray(xq)[None])[0]).all(), len(p), len(em), t1 - t0, t2 - t1

    for name, fast in (("eager chip path", True), ("CUDA-graph chip path", "graph")):
        try:
            for xq in xs[:3]: infer(xq, fast)                      # warm-up (graph capture happens here)
            rs = [infer(xq, fast) for xq in xs]
        except Exception as e:
            print(f"[{kind}:bench] {name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True); continue
        ok = sum(r[0] for r in rs); np_, ne = rs[0][1], rs[0][2]
        pre = np.mean([r[3] for r in rs]); dec = np.mean([r[4] for r in rs])
        print(f"[{kind}:bench] {name}: exact {ok}/{len(rs)}; layers {gg.n_layers()}; prompt {np_} tok, emitted {ne} tok; prefill {pre*1e3:.1f} ms; "
              f"emitted-token loop {dec*1e3:.1f} ms = {ne/dec:.1f} tok/s ({dec/ne*1e3:.2f} ms/token); end-to-end {(pre+dec)*1e3:.1f} ms per inference "
              f"({1/(pre+dec):.2f} inferences/s)", flush=True)
        bad += len(rs) - ok
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    phase = sys.argv[1]; kinds = sys.argv[2:] or ["scalar", "vec1", "vec2"]
    for k in kinds:
        if phase == "exact": exact(k)
        elif phase == "general": exact_general(k)
        elif phase == "bench": bench(k)
    print("TOTAL BAD", bad)
