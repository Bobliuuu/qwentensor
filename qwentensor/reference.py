"""Bit-exact numpy interpreter of the integer IR (the golden), plus a float interpreter for the un-quantized graph.

Everything is int64 and exact.  No torch.  Batched over axis 0 (each row is independent, identical to running the
single-sample machine program per row).
"""
import numpy as np

from .ir import Graph


def encode_input(g: Graph, x):
    """Host-side encoding of RAW samples into the input node's integer vector (this is prompt construction, not machine work).
    uint:   x already float in input units -> clip(round(x / scale), 0, amax)
    onehot: x integer token ids (B, T) -> (B, T*vocab) 0/1
    posneg: x float (B, n) -> [clip(round(relu(x)/s)) ; clip(round(relu(-x)/s))]
    Float-profile graphs return float64 values (no rounding)."""
    n = g.input; x = np.asarray(x)
    if n.encode == "onehot":
        x = x.reshape(len(x), -1).astype(np.int64); T = x.shape[1]
        assert x.min() >= 0 and x.max() < n.vocab and T * n.vocab == n.out_len, "token id out of vocab / wrong length"
        out = np.zeros((len(x), n.out_len), dtype=np.int64)
        out[np.arange(len(x))[:, None], np.arange(T)[None, :] * n.vocab + x] = 1
        return out.astype(np.float64) if g.profile == "float" else out
    x = x.reshape(len(x), -1).astype(np.float64)
    if n.encode == "posneg": x = np.concatenate([np.maximum(x, 0), np.maximum(-x, 0)], 1)
    assert x.shape[1] == n.out_len, f"input has {x.shape[1]} features, graph wants {n.out_len}"
    if g.profile == "float": return x
    s = n.scale
    return np.clip(np.rint(x / s), 0, g.amax).astype(np.int64)


def _gather(x, idx):
    out = x[:, np.maximum(idx, 0)]
    if (idx < 0).any(): out = np.where(idx[None, :] < 0, 0, out)
    return out


def run(g: Graph, xq, trace=False):
    """Run an integer graph on ENCODED inputs xq (B, input.out_len) int.  Returns the output vector (B, out_len) int64
    (or, with trace=True, a dict name -> array for every node).  Semantics of matvec (docs/IR.md):
        acc = x.reshape(B, P, K) @ W + b          exact int64
        out = clip(acc >> shift, 0, amax)         act = relu_clamp  (>> is arithmetic = floor)
        out = acc >> shift                        act = linear      (signed, not machine-native)"""
    assert g.profile != "float"
    vals = {}
    for n in g.nodes:
        if n.op == "input":
            x = np.asarray(xq, dtype=np.int64)
            assert x.ndim == 2 and x.shape[1] == n.out_len and x.min() >= 0 and x.max() <= g.amax, "bad encoded input"
            vals[n.name] = x
        elif n.op == "gather":
            vals[n.name] = _gather(vals[n.inputs[0]], n.index)
        elif n.op == "concat":
            vals[n.name] = np.concatenate([vals[i] for i in n.inputs], 1)
        else:
            x = vals[n.inputs[0]]; B = len(x); K, N = n.W.shape
            acc = x.reshape(B, n.groups, K) @ n.W.astype(np.int64) + n.b.astype(np.int64)
            y = acc >> n.shift
            if n.act == "relu_clamp": y = np.clip(y, 0, g.amax)
            vals[n.name] = y.reshape(B, n.groups * N)
    return vals if trace else vals[g.output]


def predict(g: Graph, x, raw=True):
    """Convenience: raw samples -> encode -> run -> (argmax if the graph says so)."""
    y = run(g, encode_input(g, x) if raw else x)
    return y.argmax(1) if g.argmax else y        # np.argmax: lowest index on ties


def run_float(g: Graph, x, trace=False):
    """Float64 interpreter for profile='float' graphs.  x: ENCODED float input (use encode_input)."""
    assert g.profile == "float"
    vals = {}
    for n in g.nodes:
        if n.op == "input": vals[n.name] = np.asarray(x, dtype=np.float64)
        elif n.op == "gather": vals[n.name] = _gather(vals[n.inputs[0]], n.index)
        elif n.op == "concat": vals[n.name] = np.concatenate([vals[i] for i in n.inputs], 1)
        else:
            v = vals[n.inputs[0]]; B = len(v); K, N = n.W.shape
            z = v.reshape(B, n.groups, K) @ n.W + n.b
            if n.act == "relu": z = np.maximum(z, 0)
            vals[n.name] = z.reshape(B, n.groups * N)
    return vals if trace else vals[g.output]
