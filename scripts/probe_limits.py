"""Probes behind docs/LIMITS.md: push each machine bound until it breaks and print numbers.

usage: python scripts/probe_limits.py <probe> [args]
  size      build the circuit for growing machines: layers, residual dims, hidden, vocab, parameters per Qwen3.5 size
  fan       which (rows, fan, shift) machine circuits build (the requant window), and the exactness range
  bf16      integer exactness of bf16 (where counts stop being exact)
  config    per-size capacity table from the HF text configs (reads ./configs or HF)
All probes are CPU only (circuit build / arithmetic); GPU probes are in scripts/probe_limits_gpu.py.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.environ.get("QWENTPU_SRC", os.path.expanduser("~/qwentpu_src/qwentpu"))]

from qwentensor.machine.spec import Spec
from qwentensor.machine.circuit import build


def try_build(**kw):
    sp = Spec(**kw)
    t = time.time()
    try:
        g, d = build(sp)
    except Exception as e:
        return sp, None, f"{type(e).__name__}: {str(e)[:160]}", time.time() - t
    return sp, g, None, time.time() - t


def probe_size(args):
    cases = [dict(rows=16, cols=(c,), nops=n, fan=1, smax=5, ab=5, wb=5) for c, n in
             [(15, 15), (30, 30), (45, 45), (60, 60), (90, 90), (100, 100), (126, 126), (126, 128), (127, 128), (126, 129)]]
    for kw in cases:
        sp, g, err, dt = try_build(**kw)
        if g is None:
            print(f"cols={kw['cols'][0]} nops={kw['nops']}: BUILD FAILS: {err}", flush=True); continue
        print(json.dumps({"cols": sp.ncols, "nops": sp.nops, "layers": g.n_layers(), "dims": len(g.names), "nvocab": sp.nvocab,
                          **g.stats(), "build_s": round(dt, 1)}), flush=True)


# Qwen3.5 text-tower configs (HF Qwen/Qwen3.5-*, config.json text_config): hidden, intermediate, layers, key heads, value heads
SIZES = {"0.8B": (1024, 3584, 24, 16, 16), "2B": (2048, 6144, 24, 16, 16), "4B": (2560, 9216, 32, 16, 32),
         "9B": (4096, 12288, 32, 16, 32), "27B": (5120, 17408, 64, 16, 48)}
VOCAB0 = 248320


def merged_gb(sp, g, name, n_dims=None):
    """bf16 GB of the merged model: Qwen's own L layers WIDENED to the chip hidden size (zero padding stores them at full width) plus the
    chip layers plus the untied embedding and lm_head.  Calibrated against the A100 run: 0.8B, 126 columns -> 33.1 GB measured after build."""
    from qwentpu.compiler.emit import pick_layout
    h0, inter, L, nk, nv = SIZES[name]
    n = (n_dims or len(g.names)) + 1; h = pick_layout(h0, n).hidden
    per_layer = (2 * nk * 128 + nv * 128) * h + nv * 128 * h + 2 * nv * h + h * nv * 128 + 3 * inter * h
    return h, round(((L + g.n_layers()) * per_layer + 2 * (VOCAB0 + sp.nvocab) * h) * 2 / 1e9, 1)


def probe_config(args):
    """Merged-model memory of a machine with N columns on every dense Qwen3.5 size, and the largest N that fits 40 / 80 GB."""
    cache = {}
    for cols in (15, 30, 45, 60, 75, 90, 105, 115, 126):
        sp = Spec(rows=16, cols=(cols,), nops=cols, fan=1, smax=5, ab=5, wb=5); g, d = build(sp); cache[cols] = (sp, g)
        row = {"machine_cols": cols, "residual_dims": len(g.names) + 1, "chip_layers": g.n_layers(), "nvocab": sp.nvocab}
        for name in SIZES: h, gb = merged_gb(sp, g, name); row[name] = {"hidden": h, "bf16_GB": gb}
        print(json.dumps(row), flush=True)
    for name in SIZES:
        fit = {}
        for cap in (40, 80):
            ok = [c for c, (sp, g) in cache.items() if merged_gb(sp, g, name)[1] <= cap * 0.92]    # 8% headroom for state, activations and the CUDA context
            fit[f"max_cols_in_{cap}GB"] = max(ok) if ok else 0
        print(json.dumps({"model": name, **fit}), flush=True)


def probe_fan(args):
    """Largest fan * rows that builds, per (rows, shift cap smax): the 256-wide gate window of the requant gadget."""
    for rows in (4, 8, 16, 32, 64, 128):
        for smax in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12):
            best = None; first_fail = None
            for fan in range(1, 65):
                sp, g, err, dt = try_build(rows=rows, cols=(max(fan, 2),) * 1, nops=8, fan=fan, smax=smax, ab=5, wb=5)
                if g is not None: best = fan
                else:
                    first_fail = (fan, err); break
            print(json.dumps({"rows": rows, "smax": smax, "max_fan": best, "max_fan_x_rows": None if best is None else best * rows,
                              "max_inputs_per_neuron": None if best is None else best * (rows - 1),
                              "first_failure": first_fail}), flush=True)


def probe_bits(args):
    """Largest activation / weight bit widths that build: (ab, wb) grid at rows=16, fan=1, smax=5."""
    for ab in range(1, 10):
        row = []
        for wb in range(1, 10):
            sp, g, err, dt = try_build(rows=16, cols=(8,), nops=8, fan=1, smax=5, ab=ab, wb=wb)
            row.append("ok" if g is not None else "x")
            if g is None and wb == ab: first = err
        print(json.dumps({"ab": ab, **{f"wb{w + 1}": r for w, r in enumerate(row)}}), flush=True)
    for ab, wb in ((5, 5), (6, 6), (7, 7), (8, 8), (4, 8), (8, 4)):
        sp, g, err, dt = try_build(rows=16, cols=(8,), nops=8, fan=1, smax=5, ab=ab, wb=wb)
        print(json.dumps({"ab": ab, "wb": wb, "builds": g is not None, "error": err}), flush=True)


def probe_rows(args):
    """rows (registers per tile) at the maximum machine: 126 columns, fan 1, shift cap smax in {0, 4, 5}.  Weights per machine = columns x (rows - 1)."""
    for rows in (8, 16, 32, 64, 128):
        for smax in (0, 4, 5):
            sp, g, err, dt = try_build(rows=rows, cols=(126,), nops=126, fan=1, smax=smax, ab=5, wb=5)
            rec = {"rows": rows, "smax": smax, "builds": g is not None, "error": err}
            if g is not None:
                h, gb = merged_gb(sp, g, "0.8B")
                rec.update(layers=g.n_layers(), dims=len(g.names), nvocab=sp.nvocab, hidden_0_8B=h, bf16_GB_0_8B=gb, weights_per_machine=126 * (rows - 1))
            print(json.dumps(rec), flush=True)


def probe_bf16(args):
    """The arithmetic reason for WINDOW = 256: bf16 holds integers exactly only up to 2^8."""
    import torch
    bad = [n for n in range(0, 600) if float(torch.tensor(float(n)).to(torch.bfloat16)) != n]
    print(json.dumps({"largest_exact_integer_run_from_0": bad[0] - 1, "first_unrepresentable": bad[0],
                      "bf16(257)": float(torch.tensor(257.0).to(torch.bfloat16)), "bf16(255)": float(torch.tensor(255.0).to(torch.bfloat16)),
                      "bf16(256)": float(torch.tensor(256.0).to(torch.bfloat16))}))
    k = 128.0                                       # the SwiGLU unit silu(K b) * (a / K) with K = 128
    out = {}
    for b in (-300, -2, -1, 0, 1, 2, 255, 256, 257, 258, 300):
        z = torch.tensor(k * b).to(torch.bfloat16); s = torch.nn.functional.silu(z)
        out[b] = float(s) / k
    print(json.dumps({"silu(128 b)/128 in bf16": out}))



def edge_program(sp, rng, target, shift):
    """One op summing all `fan` columns (one per tile) whose exact accumulator y equals `target`: every register holds 31 except a few
    rows, weights chosen greedily so that sum(act*w) == target (rows with act 31 take multiples of 31, one row with act 1 fixes the rest)."""
    from qwentensor.machine.asm import Op, Program
    rows, fan = sp.rows, sp.tiles - 1; AM = sp.amax
    N = rows * fan; acts = [AM] * N; fix = N - 1; acts[fix] = 1
    w = [0] * N; rem = target
    for i in range(N - 1):
        w[i] = int(max(sp.wlo, min(sp.whi, round(rem / AM)))); rem -= AM * w[i]
    if not (sp.wlo <= rem <= sp.whi): return None
    w[fix] = rem
    W = {(i % rows, i // rows): w[i] for i in range(N) if w[i]}
    init = {i: acts[i] for i in range(N)}
    dest = N
    prog = Program(sp, W, {0: Op(list(range(fan)), shift, dest)}, [0], init, entry=rows - 1)
    y = sum(a * x for a, x in zip(acts, w)); assert y == target
    return prog


def probe_extremes(args):
    """Numerics at the frontier: for each buildable (rows, fan, smax) at the largest fan*rows, run the exact float64 circuit evaluator
    (every bf16 / silu / state-row exactness condition asserted) on accumulators exactly at the requant edges, against the numpy token golden."""
    import numpy as np
    from qwentensor.machine.run import IRMachine
    configs = [(16, 3, 7), (16, 5, 4), (8, 15, 3), (64, 3, 2), (128, 2, 1)] if not args else [tuple(int(x) for x in a.split(",")) for a in args]
    for cfg in configs:
        rows, fan, smax = cfg[:3]; ab, wb = cfg[3:5] if len(cfg) == 5 else (5, 5)
        sp = Spec(rows=rows, cols=(1,) * fan + (0,), ab=ab, wb=wb, nops=2, fan=fan, smax=smax)
        t = time.time(); m = IRMachine(sp); built = time.time() - t
        ok = tot = 0; unreachable = 0; fails = []
        rng = np.random.default_rng(0)
        for s in sorted({0, smax // 2, smax}):
            for tgt in [-1, 0, 1, (1 << s) - 1, 1 << s, (sp.amax << s) - 1, sp.amax << s, (sp.nout << s) - 1, sp.nout << s, sp.whi * sp.amax * (rows * fan - 1), sp.wlo * sp.amax * (rows * fan - 1)]:
                prog = edge_program(sp, rng, tgt, s)
                if prog is None: unreachable += 1; continue
                p = prog.prompt(); em = prog.expected(); tot += 1
                try:
                    pred = m.run(p, em); good = pred == em + [sp.HALT]
                except AssertionError as e:
                    good = False; fails.append(f"shift {s} target {tgt}: evaluator assertion {str(e)[:80]}")
                ok += good
                if not good and len(fails) < 3: fails.append(f"shift {s} target {tgt}")
        print(json.dumps({"rows": rows, "fan": fan, "smax": smax, "ab": ab, "wb": wb, "fan_x_rows": rows * fan, "inputs_per_neuron": fan * rows,
                          "chip_layers": m.g.n_layers(), "build_s": round(built, 1), "edge_programs": tot, "exact": ok,
                          "targets_outside_weight_range": unreachable, "failures": fails}), flush=True)


def stress_program(sp, rng, shift, p_clear=0.1):
    """Adversarial accumulators for the digit mux: all activations at amax, every weight w = wmax with each of its non-sign bits cleared with
    probability p_clear, so every bit-plane count S[a][b] sits near fan*rows with arbitrary parity (bf16 holds odd integers only up to 256)."""
    from qwentensor.machine.asm import Op, Program
    rows, fan = sp.rows, sp.tiles - 1; N = rows * fan
    W = {}
    for i in range(N):
        w = sp.whi
        for b in range(sp.wb - 1):
            if rng.random() < p_clear: w &= ~(1 << b)
        if w: W[(i % rows, i // rows)] = int(w)
    init = {i: sp.amax for i in range(N)}
    return Program(sp, W, {0: Op(list(range(fan)), shift, N)}, [0], init, entry=rows - 1)


def probe_stress(args):
    """Is the circuit actually exact at (rows, fan, smax)?  builds != exact: the build only checks fan*rows <= 256 and the staircase window,
    while the shift mux materialises digit values up to min(ab, wb-1) * fan*rows in bf16 (integers exact only up to 256).
    usage: probe_stress rows,fan,smax[,ab,wb] ...   prints per config: builds, trials, exact, first failure."""
    import numpy as np
    from qwentensor.machine.run import IRMachine
    for a in args:
        cfg = [int(x) for x in a.split(",")]; rows, fan, smax = cfg[:3]; ab, wb = cfg[3:5] if len(cfg) == 5 else (5, 5)
        sp = Spec(rows=rows, cols=(1,) * fan + (0,), ab=ab, wb=wb, nops=2, fan=fan, smax=smax)
        try: m = IRMachine(sp)
        except Exception as e:
            print(json.dumps({"rows": rows, "fan": fan, "smax": smax, "ab": ab, "wb": wb, "F": rows * fan, "builds": False, "error": f"{type(e).__name__}: {str(e)[:100]}"}), flush=True); continue
        rng = np.random.default_rng(1); ok = tot = 0; first = None
        for tr in range(int(os.environ.get("TRIALS", 10))):
            for s in sorted({0, smax}):
                prog = stress_program(sp, rng, s, p_clear=(0.0, 0.05, 0.1, 0.2)[tr % 4]); p = prog.prompt(); em = prog.expected(); tot += 1
                try: good = m.run(p, em) == em + [sp.HALT]
                except AssertionError as e:
                    good = False; first = first or str(e)[:90]
                ok += good
                if not good and first is None: first = "token mismatch"
        print(json.dumps({"rows": rows, "fan": fan, "smax": smax, "ab": ab, "wb": wb, "F": rows * fan, "builds": True, "trials": tot, "exact": ok, "first_failure": first}), flush=True)


def probe_bitmem(args):
    """Bit-width trade at the full machine: ab, wb set the vocabulary (2^ab ACT tokens per register, 2^wb CELL tokens per weight cell)
    and the residual width (ab*wb count channels per column), so memory, not arithmetic, caps the bit width first."""
    from qwentpu.compiler.emit import pick_layout
    for ab, wb in [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]:
        for cols in (45, 126):
            sp = Spec(rows=16, cols=(cols,), nops=cols, fan=1, smax=5, ab=ab, wb=wb)
            try: g, d = build(sp)
            except Exception as e:
                print(json.dumps({"ab": ab, "wb": wb, "cols": cols, "error": str(e)[:100]})); continue
            n = len(g.names) + 1; nchip = g.n_layers()
            row = {"ab": ab, "wb": wb, "cols": cols, "residual_dims": n, "chip_layers": nchip, "nvocab": sp.nvocab}
            for name in ("0.8B", "2B", "4B"):
                h, gb = merged_gb(sp, g, name); row[name] = {"hidden": h, "bf16_GB": gb}
            print(json.dumps(row), flush=True)


def probe_maxpool(args):
    """MaxPool needs no new circuit: max(a, b) = b + relu(a - b) is two matvec layers of the existing machine (docs/IR.md section 3).
    Exhaustive over all 32 x 32 pairs against numpy, and a sample through the circuit evaluator."""
    import numpy as np
    from qwentensor import reference as R
    from qwentensor.ir import Graph, Node
    from qwentensor.machine.run import IRMachine
    from qwentensor.machine.sched import compile_graph
    nodes = [Node("x", "input", [], 2),
             Node("fc0", "matvec", ["x"], 2, W=np.array([[1, 0], [-1, 1]]), b=np.zeros(2, int), shift=0),    # [relu(a - b), b]
             Node("fc1", "matvec", ["fc0"], 1, W=np.array([[1], [1]]), b=np.zeros(1, int), shift=0)]        # relu(a - b) + b
    g = Graph(nodes, "fc1", ab=5, wb=5, profile="v2.1")
    pairs = np.array([(a, b) for a in range(32) for b in range(32)])
    out = np.array([R.run(g, p[None])[0][0] for p in pairs])
    print(json.dumps({"pairs": len(pairs), "reference_equals_max": bool((out == pairs.max(1)).all())}), flush=True)
    comp = compile_graph(g, rows=16); m = IRMachine(comp.spec); ok = 0; n = 0
    for a, b in [(0, 0), (31, 0), (0, 31), (31, 31), (5, 17), (17, 5), (30, 31), (31, 30), (12, 12)]:
        prog = comp.program(np.array([a, b])); pr = prog.prompt(); em = prog.expected(); pred = m.run(pr, em)
        n += 1; ok += pred == em + [comp.spec.HALT] and int(comp.decode(em)[0]) == max(a, b)
    print(json.dumps({"circuit_sample": n, "circuit_equals_max": ok, "ops": len(comp.neurons), "columns": comp.spec.ncols}), flush=True)


def probe_signed(args):
    """Signed outputs need no new circuit either: offset binary.  out = clip(y + 16, 0, 31) represents y in [-16, 15] (saturating beyond);
    the 16 is two constant rows (bias_rows = 2).  Exhaustive over a - b for all 32 x 32 pairs on the numpy golden, a sample on the circuit."""
    import numpy as np
    from qwentensor import reference as R
    from qwentensor.ir import Graph, Node
    from qwentensor.machine.run import IRMachine
    from qwentensor.machine.sched import compile_graph
    nodes = [Node("x", "input", [], 2), Node("fc0", "matvec", ["x"], 1, W=np.array([[1], [-1]]), b=np.array([16]), shift=0)]
    g = Graph(nodes, "fc0", ab=5, wb=5, profile="v2.1", bias_rows=2)
    pairs = np.array([(a, b) for a in range(32) for b in range(32)])
    out = np.array([R.run(g, p[None])[0][0] for p in pairs])
    want = np.clip(pairs[:, 0] - pairs[:, 1], -16, 15) + 16
    print(json.dumps({"pairs": len(pairs), "reference_equals_clip(a-b,-16,15)+16": bool((out == want).all())}), flush=True)
    comp = compile_graph(g, rows=16); m = IRMachine(comp.spec); ok = n = 0
    for a, b in [(0, 31), (31, 0), (5, 17), (17, 5), (20, 20), (31, 31)]:
        prog = comp.program(np.array([a, b])); pr = prog.prompt(); em = prog.expected()
        n += 1; ok += m.run(pr, em) == em + [comp.spec.HALT] and int(comp.decode(em)[0]) == int(np.clip(a - b, -16, 15)) + 16
    print(json.dumps({"circuit_sample": n, "circuit_exact": ok}), flush=True)

PROBES = {"signed": probe_signed, "maxpool": probe_maxpool, "bitmem": probe_bitmem, "stress": probe_stress, "extremes": probe_extremes, "size": probe_size, "config": probe_config, "fan": probe_fan, "bits": probe_bits, "rows": probe_rows, "bf16": probe_bf16}
if __name__ == "__main__":
    PROBES[sys.argv[1]](sys.argv[2:])
