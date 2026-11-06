"""Run the executor on the real merged Qwen3.5 (A100).  Host = generate(): argmax; append.  Everything else is in the model.
usage: python scripts/gpu_run.py <phase>   phase: digits | general | residual | throughput | stream | wide | wide784
env: QWEN_DIR = checkpoint dir (required), QWENTPU_SRC = qwenTPU repo, TAG = label printed in the log"""
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
from qwentensor.machine.pages import compile_paged
from qwentensor.machine.spec import Spec

SRC = os.environ.get("QWEN_DIR") or sys.exit("set QWEN_DIR to a Qwen3.5 checkpoint directory (and CUDA)")
TAG = os.environ.get("TAG", os.path.basename(SRC.rstrip("/")))
DATA = os.path.join(HERE, "..", "qwentensor", "machine", "data")
bad = 0


def rand_graph(rng, dims, shifts, ab=5, wb=5, residual=False, bias_rows=1):
    nodes = [Node("x", "input", [], dims[0])]; prev, plen = "x", dims[0]
    for i, (n, sh) in enumerate(zip(dims[1:], shifts)):
        W = rng.integers(-3, 4, (plen, n)); b = rng.integers(-8, 8, n)
        nodes.append(Node(f"fc{i}", "matvec", [prev], n, W=W, b=b, shift=sh)); prev, plen = f"fc{i}", n
    return Graph(nodes, prev, ab=ab, wb=wb, profile="v2.1", bias_rows=bias_rows)


def residual_graph(rng, w=7, ab=5, wb=5):
    nodes = [Node("x", "input", [], w)]
    nodes.append(Node("fc0", "matvec", ["x"], w, W=rng.integers(-3, 4, (w, w)), b=rng.integers(-8, 8, w), shift=3))
    nodes.append(Node("cat", "concat", ["fc0", "x"], 2 * w))
    W = np.concatenate([rng.integers(-3, 4, (w, w)), np.zeros((w, w), int)])
    for i in range(w): W[w + i, i] = 8                                # identity rows for the residual input x (scaled by 2^shift)
    nodes.append(Node("fc1", "matvec", ["cat"], w, W=W, b=rng.integers(-8, 8, w), shift=3))
    return Graph(nodes, "fc1", ab=ab, wb=wb, profile="v2.1")


def run_programs(tag, sp, items, readouts=("code",), model_cache={}):
    """items: [(name, comp, [xq...])].  ONE compiled model for all of them."""
    global bad
    install(sp)
    t0 = time.time()
    tc, sd, v0, lay, g = build_merged(SRC, mute_qwen=False)
    model = instantiate(tc, sd, "cuda")
    print(f"[{tag}] model {TAG} ({SRC})\n[{tag}] {sp}\n[{tag}] chip layers {g.n_layers()} hidden {lay.hidden} vocab {sp.nvocab} build {time.time()-t0:.0f}s", flush=True)
    for name, comp, xs in items:
        ok = 0; n = 0; t1 = time.time(); ntok = 0
        for xq in xs:
            prog = comp.program(xq); p = prog.prompt(); want = prog.expected()
            ref = R.run(comp.graph, np.asarray(xq)[None])[0]
            for ro in readouts:
                sess = TpuSession(model, v0, fast=True, readout=ro)
                em = generate(sess.feed, p, sp.HALT)
                good = em == want and (comp.decode(em) == ref).all()
                ok += good; n += 1; ntok += len(em)
                if not good: print("MISMATCH", name, ro, flush=True)
        bad += n - ok
        print(f"[{tag}] {name}: {ok}/{n} exact (emitted tokens == numpy token golden AND outputs == reference.run); prompt {len(p)} tok, emitted {len(want)}; {time.time()-t1:.1f}s", flush=True)
    del model; torch.cuda.empty_cache()
    return g


def run_stream(tag, g, z, n, readouts, max_cols=45, max_ops=60):
    """Paged execution of a network larger than one machine: every page is a fresh prompt on the SAME merged model; the host only
    builds prompts, runs argmax; append, and copies the integer activations between pages (docs/STREAMING.md)."""
    global bad
    pc = compile_paged(g, rows=16, max_cols=max_cols, max_ops=max_ops); sp = pc.spec; st = pc.stats()
    install(sp); t0 = time.time()
    tc, sd, v0, lay, gg = build_merged(SRC, mute_qwen=False)
    model = instantiate(tc, sd, "cuda")
    print(f"[{tag}] {sp}\n[{tag}] {st['neurons']} neurons in {st['pages']} pages {st['page_neurons']}; chip layers {gg.n_layers()} hidden {lay.hidden} "
          f"vocab {sp.nvocab} build {time.time()-t0:.0f}s", flush=True)
    for ro in readouts:
        ok = 0; pages_bad = 0; npages = 0; ntok = 0; t1 = time.time(); plens = []
        for i in range(n):
            xq = z["xq"][i]; ref = z["ref"][i]; page_bad = [0]

            def exec_page(prog):
                em = generate(TpuSession(model, v0, fast=True, readout=ro).feed, prog.prompt(), sp.HALT)
                if em != prog.expected(): page_bad[0] += 1
                return em
            y, sizes = pc.run(xq, exec_page)
            npages += len(sizes); pages_bad += page_bad[0]; ntok += sum(b for _, b in sizes); plens += [a for a, _ in sizes]
            good = bool((y == ref).all()) and page_bad[0] == 0; ok += good
            if not good: print("MISMATCH", tag, ro, "sample", i, flush=True)
        bad += n - ok
        print(f"[{tag}] readout {ro}: {ok}/{n} digits exact end to end (every page's emitted tokens == numpy token golden AND final outputs == reference.run); "
              f"{npages} pages run, {pages_bad} page mismatches; prompt {min(plens)}..{max(plens)} tok, {ntok} tokens emitted; {time.time()-t1:.1f}s", flush=True)
    del model; torch.cuda.empty_cache()


phase = sys.argv[1]
if phase == "digits":
    g = Graph.load(os.path.join(DATA, "digits_v21.json")); comp = compile_graph(g, rows=16)
    z = np.load(os.path.join(DATA, "digits_samples_v21.npz")); xs = z["xq"][:20]
    gm = run_programs("digits", comp.spec, [("digits MLP 15-15-10 (u5s5, shifts 2,4)", comp, list(xs))], readouts=("code", "lm_head"))
    out = np.stack([R.run(g, x[None])[0] for x in xs]); print("argmax acc on these 20:", (out.argmax(1) == z["y"][:20]).mean())
elif phase == "general":
    sp = Spec(rows=16, cols=(15, 12, 8), ab=5, wb=5, nops=40, fan=2, smax=7)
    rng = np.random.default_rng(1)
    gd = Graph.load(os.path.join(DATA, "digits_v21.json"))
    graphs = [("digits MLP 15-15-10", gd), ("random MLP 20-8-4 (K=20 spans 2 tiles)", rand_graph(rng, [20, 8, 4], [3, 2])),
              ("random 4-layer 8-5-5-5-4", rand_graph(rng, [8, 5, 5, 5, 4], [2, 2, 3, 1], bias_rows=1)),
              ("residual block (concat, identity rows)", residual_graph(rng))]
    items = []
    z = np.load(os.path.join(DATA, "digits_samples_v21.npz"))
    for name, gr in graphs:
        try: comp = compile_graph(gr, rows=16, spec_override=sp)
        except DoesNotFit as e: print("SKIP", name, e); continue
        xs = [x for x in z["xq"][:6]] if gr is gd else [rng.integers(0, 32, gr.input.out_len) for _ in range(6)]
        items.append((name, comp, xs))
    run_programs("general", sp, items, readouts=("code",))
elif phase == "residual":
    g = Graph.load(os.path.join(HERE, "data", "residual_v21.json")); comp = compile_graph(g, rows=16)
    z = np.load(os.path.join(HERE, "data", "residual_samples_v21.npz")); xs = z["xq"][:20]
    run_programs("residual", comp.spec, [("residual MLP 15-10-res10-10 (u5s5, concat + identity rows)", comp, list(xs))], readouts=("code", "lm_head"))
    out = np.stack([R.run(g, x[None])[0] for x in xs]); print("argmax acc on these 20:", (out.argmax(1) == z["y"][:20]).mean())
elif phase == "throughput":
    g = Graph.load(os.path.join(DATA, "digits_v21.json")); comp = compile_graph(g, rows=16); sp = comp.spec
    z = np.load(os.path.join(DATA, "digits_samples_v21.npz")); xs = list(z["xq"][:20])
    install(sp)
    tc, sd, v0, lay, gg = build_merged(SRC, mute_qwen=False)
    model = instantiate(tc, sd, "cuda")
    print(f"[throughput] model {TAG} {sp}", flush=True)
    sync = torch.cuda.synchronize

    def infer(xq, fast):
        prog = comp.program(xq); p = prog.prompt(); want = prog.expected()
        sess = TpuSession(model, v0, fast=fast, readout="code")
        sync(); t0 = time.perf_counter()
        nxt = sess.feed(list(p))[-1]; sync(); t1 = time.perf_counter()
        em = []
        while nxt != sp.HALT:
            em.append(nxt); nxt = sess.feed([nxt])[-1]
        sync(); t2 = time.perf_counter()
        return em == want and (comp.decode(em) == R.run(comp.graph, np.asarray(xq)[None])[0]).all(), len(p), len(em), t1 - t0, t2 - t1

    for name, fast in (("eager chip path", True), ("CUDA-graph chip path", "graph")):
        try:
            for xq in xs[:3]: infer(xq, fast)                      # warm-up (graph capture happens here)
            rs = [infer(xq, fast) for xq in xs]
        except Exception as e:
            print(f"[throughput] {name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True); continue
        ok = sum(r[0] for r in rs); np_, ne = rs[0][1], rs[0][2]
        pre = np.mean([r[3] for r in rs]); dec = np.mean([r[4] for r in rs])
        print(f"[throughput] {name}: exact {ok}/{len(rs)}; prompt {np_} tok, emitted {ne} tok; prefill {pre*1e3:.1f} ms; emitted-token loop {dec*1e3:.1f} ms = {ne/dec:.1f} tok/s ({dec/ne*1e3:.2f} ms/token); end-to-end {(pre+dec)*1e3:.1f} ms per inference ({1/(pre+dec):.2f} inf/s)", flush=True)
        bad += len(rs) - ok
elif phase == "stream":
    n = int(os.environ.get("N", 10)); want = os.environ.get("NETS", "residual,deep,cnn").split(",")
    titles = {"residual": "residual MLP 44-20-[20+res]-20-10", "deep": "deep MLP 44-44-44-44-10", "cnn": "strided CNN 1x8x8 -> conv4 -> conv8 s2 -> conv8 s2 -> fc10"}
    for name in want:
        title = titles[name]
        g = Graph.load(os.path.join(DATA, f"stream_{name}.json")); z = np.load(os.path.join(DATA, f"stream_samples_{name}.npz"))
        print(f"[stream] {title}: integer accuracy of reference.run on all {len(z['y'])} held-out digits = {(z['ref'].argmax(1) == z['y']).mean():.4f}", flush=True)
        run_stream(f"stream:{name}", g, z, n, readouts=("code", "lm_head"))
        print(f"[stream] argmax accuracy of the reference outputs on these {n} digits: {(z['ref'][:n].argmax(1) == z['y'][:n]).mean():.2f}", flush=True)
elif phase == "wide":
    n = int(os.environ.get("N", 10)); want = os.environ.get("NETS", "deep,residual,cnn").split(",")
    titles = {"deep": "deep MLP 64-96-64-32-10", "residual": "residual MLP 64-32-[32+res]-10", "cnn": "CNN conv4 -> conv8 -> AvgPool -> Linear(128->10) (512-wide Linear)"}
    for name in want:
        title = titles[name]
        g = Graph.load(os.path.join(DATA, f"wide_{name}.json")); z = np.load(os.path.join(DATA, f"wide_samples_{name}.npz"))
        print(f"[wide] {title}: integer accuracy of reference.run on all {len(z['y'])} held-out digits = {(z['ref'].argmax(1) == z['y']).mean():.4f}", flush=True)
        run_stream(f"wide:{name}", g, z, n, readouts=("code", "lm_head"))
        print(f"[wide] argmax accuracy of the reference outputs on these {n} digits: {(z['ref'][:n].argmax(1) == z['y'][:n]).mean():.2f}", flush=True)
elif phase == "wide784":
    # a 784-wide first layer (53 tiles, 18 groups).  There is no 784-input dataset offline: a seeded random 784-8-4 network, the check is bit-exactness vs reference.run.
    n = int(os.environ.get("N", 3)); rng = np.random.default_rng(784)
    g = rand_graph(rng, [784, 8, 4], [6, 5])
    for nd in g.matvecs(): nd.W = rng.integers(1, 4, nd.W.shape) * rng.choice([-1, 1], nd.W.shape)
    xq = rng.integers(0, 32, (n, 784)); ref = np.stack([R.run(g, x[None])[0] for x in xq])
    print(f"[wide784] seeded random 784-8-4 network, {n} random inputs; reference outputs {ref.tolist()}", flush=True)
    run_stream("wide784", g, dict(xq=xq, ref=ref), n, readouts=("code", "lm_head"), max_cols=60, max_ops=8)
print("TOTAL BAD", bad)
