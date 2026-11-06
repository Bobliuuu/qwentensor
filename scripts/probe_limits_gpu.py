"""Largest machines on the real merged Qwen3.5 (A100): fill every column / op slot the circuit allows and check bit-exactness.
usage: python scripts/probe_limits_gpu.py <case>
  wide126   rows=16: 8 layers of 15 neurons + a layer of 6 = 126 neurons = 126 physical columns (every neuron reads one full tile)
  deep126   rows=16: a chain of 126 one-input neurons = depth 126 (126 columns, 126 op slots, 126 sequential READs)
  rows32    rows=32: 4 layers of 31 neurons = 124 neurons x 31 inputs (about 3.8K weights, the most one A100-40GB holds)
  wide63    rows=64: a layer of 63 neurons x 63 inputs and a layer of 20 neurons x 63 inputs (shifts <= 4: the widest exact neuron with a shift)
env: QWEN_DIR = checkpoint dir, TAG = label printed in the log, QWENTPU_SRC = qwenTPU repo, N = inputs per case (default 6)"""
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
from qwentensor.machine.sched import compile_graph

SRC = os.environ.get("QWEN_DIR") or sys.exit("set QWEN_DIR to a Qwen3.5 checkpoint directory (and CUDA)")
TAG = os.environ.get("TAG", os.path.basename(SRC.rstrip("/")))
N = int(os.environ.get("N", 6))


def make_net(rng, dims, wspan, bias_span=8, cap=15):
    """Random integer MLP; per-layer shift = smallest that keeps <= 5% of the outputs saturated on random inputs (so the
    activations stay alive through every layer instead of collapsing to 0 or 31)."""
    X = rng.integers(0, 32, (256, dims[0])); nodes = [Node("x", "input", [], dims[0])]; prev = "x"; stats = []
    for i, n in enumerate(dims[1:]):
        W = rng.integers(-wspan, wspan + 1, (dims[i], n)); b = rng.integers(-bias_span, bias_span, n); y = X @ W + b
        for s in range(0, cap + 1):
            out = np.clip(y >> s, 0, 31)
            if (out >= 31).mean() <= 0.05: break
        nodes.append(Node(f"fc{i}", "matvec", [prev], n, W=W, b=b, shift=s)); prev = f"fc{i}"; X = out
        stats.append((s, round(float((out > 0).mean()), 2)))
    return Graph(nodes, prev, ab=5, wb=5, profile="v2.1"), stats


def case_graph(case, rng):
    if case == "wide126": return make_net(rng, [15] + [15] * 8 + [6], 3), 16
    if case == "deep126":                       # x <- clip(x + b_k, 0, 31) for 126 steps: stays alive, depends on every step
        nodes = [Node("x", "input", [], 1)]; prev = "x"
        for k in range(126):
            nodes.append(Node(f"fc{k}", "matvec", [prev], 1, W=np.array([[1]]), b=np.array([int(rng.choice([-1, 0, 1, 1]))]), shift=0)); prev = f"fc{k}"
        return (Graph(nodes, prev, ab=5, wb=5, profile="v2.1"), [(0, 1.0)]), 16
    if case == "rows32": return make_net(rng, [31] * 5, 2), 32
    if case == "wide63": return make_net(rng, [63, 63, 20], 2, cap=4), 64
    raise SystemExit(f"unknown case {case}")


case = sys.argv[1]
rng = np.random.default_rng(7)
(g, stats), rows = case_graph(case, rng)
comp = compile_graph(g, rows=rows); sp = comp.spec
print(f"[{case}] {sp}\n[{case}] neurons {len(comp.neurons)} ops {sp.nops} columns {sp.ncols} tiles {sp.tiles} fan {sp.fan} smax {sp.smax} "
      f"weights placed {sum(1 for _ in comp.W)}; (shift, fraction of outputs > 0) per layer {stats}", flush=True)
install(sp)
t0 = time.time()
tc, sd, v0, lay, gg = build_merged(SRC, mute_qwen=False)
model = instantiate(tc, sd, "cuda")
mem = torch.cuda.max_memory_allocated() / 1e9
print(f"[{case}] model {TAG}: chip layers {gg.n_layers()} hidden {lay.hidden} vocab {sp.nvocab} build {time.time()-t0:.0f}s; GPU memory after build {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
xs = [rng.integers(0, 32, comp.graph.input.out_len) for _ in range(N)]
ok = 0; bad = 0; t1 = time.time()
for xq in xs:
    prog = comp.program(xq); p = prog.prompt(); want = prog.expected(); ref = R.run(comp.graph, np.asarray(xq)[None])[0]
    sess = TpuSession(model, v0, fast=True, readout="code")
    em = generate(sess.feed, p, sp.HALT)
    good = em == want and (comp.decode(em) == ref).all()
    ok += good; bad += not good
    if not good:
        first = next((i for i, (a, b) in enumerate(zip(em, want)) if a != b), min(len(em), len(want)))
        print(f"MISMATCH first differing emitted token {first} of {len(want)}", flush=True)
print(f"[{case}] {ok}/{len(xs)} exact (emitted tokens == numpy token golden AND outputs == reference.run); prompt {len(p)} tok, emitted {len(want)} tok; "
      f"peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.1f} GB; {time.time()-t1:.0f}s", flush=True)
print(f"[{case}] last-layer outputs of the first input: {comp.decode(want)[:8]}...; reference nonzero outputs {int((R.run(comp.graph, np.asarray(xs[0])[None])[0] > 0).sum())}/{len(R.run(comp.graph, np.asarray(xs[0])[None])[0])}")
print("TOTAL BAD", bad)
