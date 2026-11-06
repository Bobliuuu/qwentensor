"""Layer-skip probe and benchmark on the real merged Qwen3.5 (A100).  docs/LAYER_SKIP.md.
usage: python scripts/gpu_skip.py <phase>      phase: probe | exact | bench
env: QWEN_DIR (checkpoint, default 0.8B), QWENTPU_SRC, TAG, N (inputs per program)"""
import json, os, sys, time
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.environ.get("QWENTPU_SRC", os.path.join(HERE, "..", "..", "qwentpu_src"))]
import numpy as np, torch
from qwentpu.merge import build_merged, instantiate
from qwentpu.driver import TpuSession, ChipPath, GraphedChip
from qwentensor import reference as R
from qwentensor.ir import Graph, Node
from qwentensor.machine.gpu import install
from qwentensor.machine.run import generate
from qwentensor.machine.sched import compile_graph, DoesNotFit
from qwentensor.machine.spec import Spec
from qwentensor.machine import skip as S
from qwentensor.machine.skip import SkipSession, SkipChip, SkipGraphedChip

SRC = os.environ.get("QWEN_DIR", "/home/ubuntu/qwentpu_gpu/q08")
TAG = os.environ.get("TAG", os.path.basename(SRC.rstrip("/")))
DATA = os.path.join(HERE, "..", "qwentensor", "machine", "data")
N = int(os.environ.get("N", 4))
bad = 0


def rand_graph(rng, dims, shifts, ab=5, wb=5, bias_rows=1):
    nodes = [Node("x", "input", [], dims[0])]; prev, plen = "x", dims[0]
    for i, (n, sh) in enumerate(zip(dims[1:], shifts)):
        W = rng.integers(-3, 4, (plen, n)); b = rng.integers(-8, 8, n)
        nodes.append(Node(f"fc{i}", "matvec", [prev], n, W=W, b=b, shift=sh)); prev, plen = f"fc{i}", n
    return Graph(nodes, prev, ab=ab, wb=wb, profile="v2.1", bias_rows=bias_rows)


def programs(case):
    """[(name, comp, [xq...])] sharing ONE Spec."""
    z = np.load(os.path.join(DATA, "digits_samples_v21.npz")); gd = Graph.load(os.path.join(DATA, "digits_v21.json"))
    if case == "digits":
        comp = compile_graph(gd, rows=16); return comp.spec, [("digits MLP 15-15-10", comp, list(z["xq"][:N]))]
    if case == "general":
        sp = Spec(rows=16, cols=(15, 12, 8), ab=5, wb=5, nops=40, fan=2, smax=7); rng = np.random.default_rng(1); items = []
        for name, gr in [("digits MLP 15-15-10", gd), ("random MLP 20-8-4", rand_graph(rng, [20, 8, 4], [3, 2])),
                         ("random 4-layer 8-5-5-5-4", rand_graph(rng, [8, 5, 5, 5, 4], [2, 2, 3, 1]))]:
            comp = compile_graph(gr, rows=16, spec_override=sp)
            xs = list(z["xq"][:N]) if gr is gd else [rng.integers(0, 32, gr.input.out_len) for _ in range(N)]
            items.append((name, comp, xs))
        return sp, items
    if case == "residual":
        g = Graph.load(os.path.join(HERE, "data", "residual_v21.json")); comp = compile_graph(g, rows=16)
        zz = np.load(os.path.join(HERE, "data", "residual_samples_v21.npz"))
        return comp.spec, [("residual MLP 15-10-res10-10", comp, list(zz["xq"][:N]))]
    raise SystemExit(f"unknown case {case}")


def build(sp):
    install(sp)
    tc, sd, v0, lay, g = build_merged(SRC, mute_qwen=False)
    model = instantiate(tc, sd, "cuda")
    return model, v0, lay, g


def tclass(sp, t):
    if t == sp.HALT: return "HALT"
    if sp.ACT0 <= t < sp.READ0: return "ACT"
    if sp.READ0 <= t < sp.CELL0: return "READ"
    return "other"


def probe(case):
    sp, items = programs(case)
    model, v0, lay, g = build(sp)
    first = ChipPath(model, v0).first
    nl = model.config.num_hidden_layers - first
    st = S.static_stats(model, first)
    print(f"[probe:{case}] {sp}\n[probe:{case}] chip layers {nl} hidden {lay.hidden} vocab {sp.nvocab}", flush=True)
    print(f"[probe:{case}] static (weights only): heads used {sum(s['heads'] for s in st)}/{sum(s['of'] for s in st)}, "
          f"mlp units used {sum(s['units'] for s in st)}/{sum(s['inter'] for s in st)}, conv channels reading history "
          f"{sum(s['hist_channels'] for s in st)}/{sum(s['conv_channels'] for s in st)}", flush=True)
    for s in st: print(f"[probe:{case}]   layer {s['layer']}: heads {s['heads']}/{s['of']} units {s['units']}/{s['inter']} hist_channels {s['hist_channels']}", flush=True)
    pr = S.Probe(model, first)
    for name, comp, xs in items:
        for xq in xs:
            prog = comp.program(xq); p = prog.prompt(); want = prog.expected()
            sess = TpuSession(model, v0, fast=True, readout="code")

            def feed(toks, _f=sess.feed):
                if len(toks) == 1: pr.cur = tclass(sp, toks[0])
                else: pr.cur = "prompt"
                return _f(toks)
            em = generate(feed, p, sp.HALT); assert em == want
    pr.close()
    rows = [r for r in pr.rows if r["tok"] != "prompt"]
    print(f"[probe:{case}] replica_layer == HF layer (bitwise): {pr.replica_ok} ok, {pr.replica_bad} mismatches over {pr.replica_ok + pr.replica_bad} layer steps", flush=True)
    by = defaultdict(list)
    for r in rows: by[(r["tok"], r["layer"])].append(r)
    toks = sorted({r["tok"] for r in rows}); steps = {t: len([r for r in rows if r["tok"] == t and r["layer"] == 0]) for t in toks}
    print(f"[probe:{case}] decode steps by class {steps}", flush=True)
    for t in toks:
        tot = [r for r in rows if r["tok"] == t]
        n = len(tot)
        dorm = sum(r["attn_dormant"] and r["mlp_dormant"] for r in tot) / n
        pas = sum(r["passthrough"] for r in tot) / n
        adorm = sum(r["attn_dormant"] for r in tot) / n; mdorm = sum(r["mlp_dormant"] for r in tot) / n
        heads = np.mean([r["heads"] for r in tot]); mlp = np.mean([r["mlp"] for r in tot])
        print(f"[probe:{case}] class {t}: layer-steps {n}; attn dormant {adorm:.1%}, mlp dormant {mdorm:.1%}, whole layer dormant {dorm:.1%}, "
              f"output == input {pas:.1%}; live heads/layer {heads:.2f}, live mlp units/layer {mlp:.2f}", flush=True)
    for L in range(nl):
        tot = [r for r in rows if r["layer"] == L]
        print(f"[probe:{case}]   layer {L}: dormant {np.mean([r['attn_dormant'] and r['mlp_dormant'] for r in tot]):.1%} "
              f"live heads {np.mean([r['heads'] for r in tot]):.2f} live units {np.mean([r['mlp'] for r in tot]):.2f}", flush=True)
    allr = len(rows)
    print(f"[probe:{case}] TOTAL layer-steps {allr}; dormant whole-layer {sum(r['attn_dormant'] and r['mlp_dormant'] for r in rows)/allr:.1%}; "
          f"mean live heads/layer {np.mean([r['heads'] for r in rows]):.2f}; mean live mlp units/layer {np.mean([r['mlp'] for r in rows]):.2f}", flush=True)
    json.dump(dict(case=case, rows=rows, static=st), open(f"probe_{case}.json", "w"))


# ---------------------------------------------------------------------------------------------------------------------------
def want_ok(comp, em, want, ref):
    return em == want and (comp.decode(em) == ref).all()


def states_equal(pa, pb, first, last):
    return all(torch.equal(pa.layers[L].recurrent_states[0], pb.layers[L].recurrent_states[0]) for L in range(first, last))


def exact_case(tag, sp, items, model, v0, graph_modes=(False, True), readouts=("code", "lm_head")):
    """dense vs skip (eager, and CUDA-graph) on every program: tokens equal the numpy golden AND equal the dense path's; the
    recurrent state tensors of every chip layer are bit-equal after the run."""
    global bad
    first = ChipPath(model, v0).first; last = model.config.num_hidden_layers
    for name, comp, xs in items:
        for ro in readouts:
            ok = n = 0; sk = dict(attn_skipped=0, attn_run=0, steps=0); t1 = time.time()
            for xq in xs:
                prog = comp.program(xq); p = prog.prompt(); want = prog.expected(); ref = R.run(comp.graph, np.asarray(xq)[None])[0]
                dense = TpuSession(model, v0, fast=True, readout=ro); em_d = generate(dense.feed, p, sp.HALT)
                good = want_ok(comp, em_d, want, ref)
                for graph in graph_modes:
                    sess = SkipSession(model, v0, ro, graph=graph); em_s = generate(sess.feed, p, sp.HALT)
                    good = good and em_s == em_d and want_ok(comp, em_s, want, ref) and states_equal(dense.past, sess.past, first, last)
                    if not graph:
                        for k in sk: sk[k] += sess.path.stats[k]
                    del sess
                ok += good; n += 1
                if not good: print("MISMATCH", tag, name, ro, flush=True)
            bad += n - ok
            modes = " and ".join("graph" if g else "eager" for g in graph_modes)
            print(f"[{tag}] {name}, readout {ro}: {ok}/{n} exact (skip {modes}: tokens == numpy golden == dense path, outputs == reference.run, "
                  f"recurrent states bit-equal); attention half skipped on {sk['attn_skipped']} of {sk['attn_skipped'] + sk['attn_run']} "
                  f"layer-steps (eager); prompt {len(p)} tok, emitted {len(want)}; {time.time()-t1:.1f}s", flush=True)


def load_model(sp, inst=install):
    inst(sp); t0 = time.time()
    tc, sd, v0, lay, g = build_merged(SRC, mute_qwen=False); model = instantiate(tc, sd, "cuda")
    return model, v0, lay, g, time.time() - t0


def phase_exact(cases):
    global bad
    for case in cases:
        if case in ("digits", "general", "residual"):
            sp, items = programs(case); model, v0, lay, g, dt = load_model(sp)
            print(f"[exact:{case}] {TAG} {sp}\n[exact:{case}] chip layers {g.n_layers()} hidden {lay.hidden} vocab {sp.nvocab} build {dt:.0f}s", flush=True)
            exact_case(f"exact:{case}", sp, items, model, v0)
        elif case.startswith("vec"):
            from qwentensor.machine.vec import compile_vec, install_vec
            lanes = int(case[3:]); g0 = Graph.load(os.path.join(DATA, "digits_v21.json")); z = np.load(os.path.join(DATA, "digits_samples_v21.npz"))
            comp = compile_vec(g0, rows=16, lanes=lanes); sp = comp.spec; model, v0, lay, g, dt = load_model(sp, install_vec)
            print(f"[exact:{case}] {TAG} {sp}\n[exact:{case}] chip layers {g.n_layers()} hidden {lay.hidden} vocab {sp.nvocab}", flush=True)
            exact_case(f"exact:{case}", sp, [(f"digits MLP, {lanes} lane(s)", comp, list(z["xq"][:N]))], model, v0)
        elif case == "stream":
            from qwentensor.machine.pages import compile_paged
            g0 = Graph.load(os.path.join(DATA, "stream_residual.json")); z = np.load(os.path.join(DATA, "stream_samples_residual.npz"))
            pc = compile_paged(g0, rows=16, max_cols=45, max_ops=60); sp = pc.spec; model, v0, lay, g, dt = load_model(sp)
            first = ChipPath(model, v0).first; last = model.config.num_hidden_layers
            print(f"[exact:stream] {TAG} {sp}\n[exact:stream] chip layers {g.n_layers()} hidden {lay.hidden} vocab {sp.nvocab}", flush=True)
            nd = min(N, 4)
            for ro in ("code", "lm_head"):
                ok = pages = 0; t1 = time.time(); cnt = [0, 0]
                for i in range(nd):
                    xq = z["xq"][i]; ref = z["ref"][i]; page_bad = [0]

                    def exec_page(prog, _ro=ro):
                        p = prog.prompt(); want = prog.expected()
                        dense = TpuSession(model, v0, fast=True, readout=_ro); em_d = generate(dense.feed, p, sp.HALT)
                        for graph in (False, True):
                            sess = SkipSession(model, v0, _ro, graph=graph); em_s = generate(sess.feed, p, sp.HALT)
                            if em_s != em_d or em_d != want or not states_equal(dense.past, sess.past, first, last): page_bad[0] += 1
                            if not graph: cnt[0] += sess.path.stats["attn_skipped"]; cnt[1] += sess.path.stats["attn_run"]
                            del sess
                        return em_d
                    y, sizes = pc.run(xq, exec_page); pages += len(sizes)
                    good = bool((y == ref).all()) and page_bad[0] == 0; ok += good
                    if not good: print("MISMATCH stream", ro, i, flush=True)
                bad += nd - ok
                print(f"[exact:stream] residual MLP 44-20-[20+res]-20-10 paged, readout {ro}: {ok}/{nd} digits exact (every page: skip eager and graph tokens == dense == golden, states bit-equal); "
                      f"{pages} pages; attention half skipped on {cnt[0]} of {cnt[0] + cnt[1]} layer-steps (eager); {time.time()-t1:.1f}s", flush=True)
        else:
            raise SystemExit(f"unknown case {case}")
        try: del model
        except NameError: pass
        torch.cuda.empty_cache()
    print("TOTAL BAD", bad)


# ---------------------------------------------------------------------------------------------------------------------------
class Persistent:
    """One cache and one chip path for many inferences (graphs are captured once): the machine is re-loaded by zeroing the chip
    state and feeding the prompt again, which is what a fresh session does, without recapturing the graphs."""

    def __init__(self, model, v0, path_cls):
        from transformers import DynamicCache
        self.m, self.v0 = model, v0; self.past = DynamicCache(config=model.config)
        self.path = path_cls(model, v0, None, "code"); self.path.attach(self.past)

    def infer(self, p, halt):
        sync = torch.cuda.synchronize; self.path.zero(); dev = self.m.device
        sync(); t0 = time.perf_counter()
        nxt = int(self.path.tokens(torch.tensor([[self.v0 + t for t in p]], device=dev), self.past)[0, -1]); sync(); t1 = time.perf_counter()
        em = []
        while nxt != halt:
            em.append(nxt); nxt = int(self.path.tokens(torch.tensor([[self.v0 + nxt]], device=dev), self.past)[0, -1])
        sync(); t2 = time.perf_counter()
        return em, t1 - t0, t2 - t1


def eager_infer(model, v0, p, halt, path=None, skip=False, ro="code"):
    sync = torch.cuda.synchronize
    sess = SkipSession(model, v0, ro, path=path) if skip else TpuSession(model, v0, fast=True, readout=ro)
    sync(); t0 = time.perf_counter(); nxt = sess.feed(list(p))[-1]; sync(); t1 = time.perf_counter(); em = []
    while nxt != halt:
        em.append(nxt); nxt = sess.feed([nxt])[-1]
    sync(); t2 = time.perf_counter()
    return em, t1 - t0, t2 - t1


def phase_bench(cases):
    global bad
    for case in cases:
        sp, items = programs(case); model, v0, lay, g, dt = load_model(sp)
        name, comp, xs = items[0]; xs = xs[:max(N, 6)]
        print(f"[bench:{case}] {TAG} {name}; chip layers {g.n_layers()} hidden {lay.hidden} vocab {sp.nvocab}; {len(xs)} inputs", flush=True)
        skip_path = SkipChip(model, v0, readout="code"); skip_path_lm = SkipChip(model, v0, readout="lm_head")
        pers = {"dense graph": Persistent(model, v0, GraphedChip), "skip graph": Persistent(model, v0, SkipGraphedChip)}
        runners = [("dense eager", lambda p: eager_infer(model, v0, p, sp.HALT)), ("skip eager", lambda p: eager_infer(model, v0, p, sp.HALT, skip_path, True)),
                   ("dense graph", lambda p: pers["dense graph"].infer(p, sp.HALT)), ("skip graph", lambda p: pers["skip graph"].infer(p, sp.HALT)),
                   ("dense eager lm_head readout", lambda p: eager_infer(model, v0, p, sp.HALT, ro="lm_head")),
                   ("skip eager lm_head readout", lambda p: eager_infer(model, v0, p, sp.HALT, skip_path_lm, True, "lm_head"))]
        res = {}
        for label, fn in runners:
            for xq in xs[:3]: fn(comp.program(xq).prompt())                                    # warm-up (graph capture happens here)
            rows = []
            for xq in xs:
                prog = comp.program(xq); em, tp, tl = fn(prog.prompt())
                rows.append((em == prog.expected() and (comp.decode(em) == R.run(comp.graph, np.asarray(xq)[None])[0]).all(), len(em), tp, tl))
            ok = sum(r[0] for r in rows); bad += len(rows) - ok
            ne = rows[0][1]; pre = np.mean([r[2] for r in rows]); dec = np.mean([r[3] for r in rows]); res[label] = (pre, dec, ne)
            print(f"[bench:{case}] {label}: exact {ok}/{len(rows)}; emitted {ne} tok; prefill {pre*1e3:.1f} ms; emitted-token loop {dec*1e3:.1f} ms = {ne/dec:.1f} tok/s "
                  f"({dec/ne*1e3:.2f} ms/token); end-to-end {(pre+dec)*1e3:.1f} ms per inference ({1/(pre+dec):.2f} inferences/s)", flush=True)
        for mode in ("eager", "graph", "eager lm_head readout"):
            d, k = res[f"dense {mode}"], res[f"skip {mode}"]
            print(f"[bench:{case}] {mode}: skip vs dense: loop {d[1]/k[1]:.2f}x faster, per inference {(d[0]+d[1])/(k[0]+k[1]):.2f}x faster", flush=True)
        print(f"[bench:{case}] skip counters (eager path, all inputs): {skip_path.stats}", flush=True)
        del model, pers; torch.cuda.empty_cache()
    print("TOTAL BAD", bad)


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "probe":
        for case in (sys.argv[2:] or ["digits", "general"]): probe(case)
    elif phase == "exact":
        phase_exact(sys.argv[2:] or ["digits", "general", "residual", "vec1", "vec2", "stream"])
    elif phase == "bench":
        phase_bench(sys.argv[2:] or ["digits"])
    else:
        raise SystemExit("unknown phase")
