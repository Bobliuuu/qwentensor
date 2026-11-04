"""Post-training quantization of a float IR graph to the machine's integers, plus optional STE fine-tuning (QAT).

    q = quantize(float_graph, X_calib, QConfig(profile="v2.1"))      # integer Graph, validated
    q = qat(q, X_train, y_train, epochs=40)                           # optional: STE fine-tune the integer weights

Scheme (docs/IR.md section 4).  Every tensor element is `scale * q`, q in 0..amax.  For a matvec the layer's output scale `s_o`
and shift `sh` are chosen by a SEARCH on calibration data (sequentially, using the already-quantized upstream activations): for each
candidate (s_o, sh) the integer layer is simulated exactly and the candidate with the smallest output error against the float layer
(relu, evaluated on the dequantized quantized inputs) wins.  The float scale is folded into the integer weights
   Wq = clip(round(Wf * s_in * 2^sh / s_o)),   bq = round(bf * 2^sh / s_o) (+ 2^(sh-1) so the floor shift rounds to nearest)
No per-channel scales, no zero points (activations are post-ReLU, so unsigned with zero point 0).
Profile v2.0 forces sh = 0: s_o can then only be tuned through the weights' integer range, which is why it loses accuracy.
"""
from dataclasses import dataclass

import numpy as np

from .ir import Graph, Node
from . import reference as R


@dataclass
class QConfig:
    profile: str = "v2.1"            # v2.1: shift 0..15 allowed; v2.0: shift == 0 (what the hardware runs today)
    ab: int = 5
    wb: int = 5
    bias_rows: int = 1               # bias range = bias_rows * [wlo, whi]
    input_max: int = None            # max integer level of the input (default amax); qwenTPU's digits example uses 12
    act_pct: float = 99.9            # percentile of calibration activations that maps to amax (candidate centre)
    logits: str = "relu_clamp"       # last layer: relu_clamp (machine native, logits 0..31) | linear (signed; NOT machine-native)
    search_samples: int = 512
    round_half: bool = True          # fold 2^(sh-1) into the bias so the floor shift rounds to nearest


def _pct(a, p):
    a = a[a > 0]
    return float(np.percentile(a, p)) if a.size else 1.0


def _deq(q, s): return q * s


def quantize(fg: Graph, Xcal, cfg: QConfig = None, **kw) -> Graph:
    cfg = cfg or QConfig(**kw)
    assert fg.profile == "float" and cfg.profile in ("v2.0", "v2.1") and cfg.logits in ("relu_clamp", "linear")
    amax, wlo, whi = (1 << cfg.ab) - 1, -(1 << (cfg.wb - 1)), (1 << (cfg.wb - 1)) - 1
    blo, bhi = wlo * cfg.bias_rows, whi * cfg.bias_rows
    xin = R.encode_input(fg, Xcal)                       # float (B, in_len)
    B = len(xin); rng = np.random.default_rng(0)
    sub = rng.permutation(B)[:cfg.search_samples]
    q, s, out_nodes, report = {}, {}, [], []
    for n in fg.nodes:
        if n.op == "input":
            nn_ = Node(n.name, "input", [], n.out_len, encode=n.encode, vocab=n.vocab)
            if n.encode == "onehot":
                sc = 1.0; qv = xin.astype(np.int64)
            else:
                if xin.min() < 0: raise ValueError("input has negative values: shift them, or build the graph with input_posneg=True")
                im = cfg.input_max or amax
                sc = max(float(xin.max()), 1e-12) / im
                qv = np.clip(np.rint(xin / sc), 0, amax).astype(np.int64)
            nn_.scale = np.full(n.out_len, sc); q[n.name], s[n.name] = qv, nn_.scale
        elif n.op == "gather":
            qi, si = q[n.inputs[0]], s[n.inputs[0]]; idx = n.index
            qv = qi[:, np.maximum(idx, 0)]
            if (idx < 0).any(): qv = np.where(idx[None] < 0, 0, qv)
            sv = np.where(idx >= 0, si[np.maximum(idx, 0)], si[0])
            nn_ = Node(n.name, "gather", list(n.inputs), n.out_len, index=idx.copy(), scale=sv); q[n.name], s[n.name] = qv, sv
        elif n.op == "concat":
            nn_ = Node(n.name, "concat", list(n.inputs), n.out_len, scale=np.concatenate([s[i] for i in n.inputs]))
            q[n.name] = np.concatenate([q[i] for i in n.inputs], 1); s[n.name] = nn_.scale
        else:
            nn_, qv, sv, rep = _quant_matvec(fg, n, q[n.inputs[0]], s[n.inputs[0]], cfg, sub, (amax, wlo, whi, blo, bhi))
            q[n.name], s[n.name] = qv, sv; report.append(rep)
        out_nodes.append(nn_)
    g = Graph(out_nodes, fg.output, fg.argmax, cfg.profile, cfg.ab, cfg.wb, cfg.bias_rows,
              meta=dict(fg.meta, report=report, input_max=cfg.input_max or amax))
    return g.validate()


def _quant_matvec(fg, n, xq, sx, cfg, sub, rng_):
    amax, wlo, whi, blo, bhi = rng_
    K, N = n.W.shape; P = n.groups; Bn = len(xq)
    sk = sx.reshape(P, K)
    if not np.allclose(sk, sk[0], rtol=1e-9): raise ValueError(f"{n.name}: input scales differ between the {P} positions of a grouped matvec")
    sk = sk[0]
    final = n.name == fg.output
    mode = "relu_clamp" if (n.act == "relu" or (final and cfg.logits == "relu_clamp")) else "linear"
    xs = xq[sub].reshape(len(sub), P, K).astype(np.float64)
    Wf_s = n.W * sk[:, None]                                          # weights in "real per input-int" units
    z = xs @ Wf_s + n.b
    target = np.maximum(z, 0) if mode == "relu_clamp" else z
    if mode == "relu_clamp": base = _pct(target, cfg.act_pct) / amax; js = range(-18, 19)
    else: base = max(float(np.percentile(np.abs(target), cfg.act_pct)), 1e-12) / 127.0; js = range(-6, 7)
    shifts = [0] if cfg.profile == "v2.0" else range(0, 16)
    wmax = max(float(np.abs(Wf_s).max()), 1e-300); best = None
    for j in js:
        so = base * 2.0 ** (j / 3)
        for sh in shifts:
            m = 2.0 ** sh / so
            if not (0.5 <= wmax * m <= 60): continue                  # all-zero weights or hopeless clipping
            Wq = np.clip(np.rint(Wf_s * m), wlo, whi)
            bq = np.clip(np.rint(n.b * m) + (2 ** (sh - 1) if (cfg.round_half and sh > 0) else 0), blo, bhi)
            acc = xs @ Wq + bq
            y = np.floor(acc / 2.0 ** sh)
            if mode == "relu_clamp": y = np.clip(y, 0, amax)
            err = float(np.mean((y * so - target) ** 2))
            if best is None or err < best[0]: best = (err, so, sh)
    if best is None:                                                  # degenerate: fall back to weight-limited scale
        so = wmax / whi; best = (np.nan, so, 0)
    err, so, sh = best; m = 2.0 ** sh / so
    Wq_raw = np.rint(n.W * sk[:, None] * m); Wq = np.clip(Wq_raw, wlo, whi).astype(np.int64)
    bq_raw = np.rint(n.b * m) + (2 ** (sh - 1) if (cfg.round_half and sh > 0) else 0)
    bq = np.clip(bq_raw, blo, bhi).astype(np.int64)
    node = Node(n.name, "matvec", list(n.inputs), n.out_len, W=Wq, b=bq, shift=int(sh), act=mode, groups=P,
                scale=np.full(n.out_len, so))
    # run on the whole calibration set to get the real quantized activations for downstream layers
    acc = xq.reshape(Bn, P, K) @ Wq + bq; y = acc >> sh
    if mode == "relu_clamp": y = np.clip(y, 0, amax)
    rep = dict(name=n.name, shape=[K, N], groups=P, shift=int(sh), out_scale=float(so), act=mode,
               w_clipped=float((Wq_raw != Wq).mean()), b_clipped=float((bq_raw != bq).mean()),
               w_utilisation=float(np.abs(Wq).max() / max(abs(wlo), whi)), search_mse=float(err),
               out_saturated=float((y == amax).mean()) if mode == "relu_clamp" else 0.0,
               out_zero=float((y == 0).mean()) if mode == "relu_clamp" else 0.0)
    return node, y.reshape(Bn, P * N), node.scale, rep


# ------------------------------------------------------------------------------------------------ evaluation
def accuracy(g: Graph, X, y):
    p = R.predict(g, X); p = p.argmax(1) if p.ndim == 2 else p
    return float((p == np.asarray(y)).mean())


def float_accuracy(fg: Graph, X, y):
    out = R.run_float(fg, R.encode_input(fg, X))
    return float((out.argmax(1) == np.asarray(y)).mean())


# ------------------------------------------------------------------------------------------------ QAT (STE)
def qat(g: Graph, X, y, epochs=40, lr=0.01, batch=64, leak=0.15, temp=None, seed=0, eval_every=2, verbose=False, seeds=1, init="ptq") -> Graph:
    """Runs `_qat_once` for `seeds` shuffling seeds and returns the graph with the best TRAIN accuracy."""
    runs = [_qat_once(g, X, y, epochs, lr, batch, leak, temp, seed + i, eval_every, verbose, init) for i in range(seeds)]
    return max(runs, key=lambda r: r.meta.get("qat_train_acc", 0))


def _qat_once(g, X, y, epochs, lr, batch, leak, temp, seed, eval_every, verbose, init) -> Graph:
    """STE fine-tune of the INTEGER weights/biases of a quantized graph (scales and shifts are frozen) for classification.
    Forward is the exact integer arithmetic of the IR: round-to-int weights (STE), exact accumulate, floor shift (STE), clamp with a leaky
    gradient outside [0, amax] (as qwenTPU examples/mlp/train.py).  Model selection on TRAIN accuracy via the bit-exact reference
    (the initial PTQ graph is a candidate, so QAT never returns something worse on train)."""
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X = np.asarray(X); y = np.asarray(y)
    xq = torch.tensor(R.encode_input(g, X), dtype=torch.float64); yt = torch.tensor(y)
    amax = g.amax
    mvs = g.matvecs(); P_ = {n.name: dict(W=torch.tensor(n.W, dtype=torch.float64, requires_grad=True),
                                           b=torch.tensor(n.b, dtype=torch.float64, requires_grad=True)) for n in mvs}
    if init == "random":      # qwenTPU examples/mlp/train.py style: integer-aware training from scratch (scales/shifts of g kept)
        for k, n in enumerate(mvs):
            with torch.no_grad():
                P_[n.name]["W"].copy_(torch.randn(*n.W.shape, dtype=torch.float64) * 0.5)
                P_[n.name]["b"].copy_(torch.ones(n.b.shape, dtype=torch.float64) if k < len(mvs) - 1 else torch.zeros(n.b.shape, dtype=torch.float64))
    params = [t for d in P_.values() for t in d.values()]
    opt = torch.optim.Adam(params, lr=lr)

    def rnd(p, lo, hi):
        r = torch.clamp(torch.round(p), lo, hi); return p + (r - p).detach()

    def fwd(xb):
        v = {}
        for n in g.nodes:
            if n.op == "input": v[n.name] = xb
            elif n.op == "gather":
                a = v[n.inputs[0]]; idx = torch.tensor(np.maximum(n.index, 0)); o = a[:, idx]
                v[n.name] = o * torch.tensor((n.index >= 0).astype(np.float64)) if (n.index < 0).any() else o
            elif n.op == "concat": v[n.name] = torch.cat([v[i] for i in n.inputs], 1)
            else:
                a = v[n.inputs[0]]; K, N = n.W.shape
                W = rnd(P_[n.name]["W"], g.wlo, g.whi); b = rnd(P_[n.name]["b"], g.blo, g.bhi)
                acc = a.reshape(len(a), n.groups, K) @ W + b
                t = acc / 2.0 ** n.shift
                f = t + (torch.floor(t) - t).detach()                       # floor with identity gradient
                if n.act == "relu_clamp":
                    c = torch.clamp(f, 0, amax); sur = c + leak * (f - c)  # leaky outside the linear range
                    f = c.detach() + (sur - sur.detach())
                v[n.name] = f.reshape(len(a), n.groups * N)
        return v[g.output]

    def snapshot():
        ns = []
        for n in g.nodes:
            if n.op == "matvec":
                W = np.clip(np.rint(P_[n.name]["W"].detach().numpy()), g.wlo, g.whi).astype(np.int64)
                b = np.clip(np.rint(P_[n.name]["b"].detach().numpy()), g.blo, g.bhi).astype(np.int64)
                m = Node(**{**n.__dict__, "W": W, "b": b})
            else: m = n
            ns.append(m)
        return Graph(ns, g.output, g.argmax, g.profile, g.ab, g.wb, g.bias_rows, dict(g.meta))

    best = (accuracy(g, X, y) if init == "ptq" else -1.0, g)
    with torch.no_grad(): sd = float(fwd(xq[:512]).std())
    T = temp or (4.0 if g.node(g.output).act == "relu_clamp" else max(sd / 2, 1.0))
    N = len(X)
    for ep in range(epochs):
        cur = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) * 0.9 + 0.1)
        for gr in opt.param_groups: gr["lr"] = cur
        idx = rng.permutation(N)
        for st in range(0, N, batch):
            i = torch.tensor(idx[st:st + batch])
            loss = torch.nn.functional.cross_entropy(fwd(xq[i]) / T, yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for n in mvs:
                    P_[n.name]["W"].clamp_(g.wlo - .49, g.whi + .49); P_[n.name]["b"].clamp_(g.blo - .49, g.bhi + .49)
        if ep % eval_every == eval_every - 1 or ep == epochs - 1:
            cand = snapshot(); ac = accuracy(cand, X, y)
            if verbose: print(f"  qat epoch {ep + 1}: train acc {ac:.4f} (best {best[0]:.4f})")
            if ac > best[0]: best = (ac, cand)
    out = best[1]; out.meta = dict(out.meta, qat_train_acc=best[0]); return out
