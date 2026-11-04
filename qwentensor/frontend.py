"""torch.fx frontend: nn.Module -> float IR graph (profile="float"), ready for `quant.quantize`.

    g = from_torch(model, input_shape=(1, 8, 8))            # float Graph
    g = from_torch(model, (T,), input_kind="index")         # token ids -> Embedding (one-hot matvec)

How it works.  fx values are tracked as
  U    a materialized non-negative tensor (an activation the machine holds: output of a ReLU'd matvec, or the input)
  Aff  a not-yet-materialized AFFINE function of U tensors:  sum_t term_t(U_t) + bias, term = dense (K x N) or grouped (conv).
Linear / Add / Flatten / AvgPool / Embedding keep things affine (they compose or concatenate terms, in exact float
arithmetic); ReLU (or the end of the network) materializes an Aff as ONE fused matvec node.  So Linear->Linear without a
ReLU between is a single matvec, a residual `relu(x + f(h))` is a single matvec over concat[h, x], and
Flatten / permutations never cost a gather: weights of the consumer are re-indexed.  Only Conv2d needs a gather (im2col).
Anything that is not affine and not ReLU (softmax, LayerNorm, GELU, MaxPool, ...) raises UnsupportedOpError.
"""
import operator
from dataclasses import dataclass

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from .ir import Graph, Node


class UnsupportedOpError(NotImplementedError):
    """Raised for any PyTorch op the machine IR cannot express.  The message names the op and the graph node."""


_HINTS = {
    "softmax": "exp/normalize is not an affine+ReLU op. If it is the LAST op before argmax, drop it (argmax is invariant); "
               "otherwise it needs a LUT/exp gadget (docs/IR.md, 'Not supported').",
    "log_softmax": "same as softmax; drop it when only argmax is needed.",
    "layer_norm": "needs mean/variance, a reciprocal-sqrt and a per-sample multiply; not expressible (docs/IR.md).",
    "batch_norm": "fold BatchNorm into the preceding Linear/Conv2d first (torch.nn.utils.fusion.fuse_conv_bn_eval / fuse_linear_bn_eval).",
    "gelu": "no smooth activation exists; train with ReLU, or approximate with a piecewise-linear LUT (docs/IR.md).",
    "silu": "no smooth activation exists; train with ReLU instead.",
    "sigmoid": "needs a LUT; not available in the v2 datapath. Train with ReLU, or drop it when only argmax is needed.",
    "tanh": "needs a LUT; not available in the v2 datapath.",
    "max_pool": "max is not affine; use AvgPool2d (or stride-2 Conv2d).",
    "multi_head_attention": "attention needs softmax and a data x data product (matvec weights are stationary); not supported.",
    "mul": "elementwise product of two activations is not a matvec; only multiplication by a constant (fold it into weights) is.",
    "matmul": "data x data matmul is not supported; weights must be constants (nn.Linear).",
    "dropout": None,
}


def _unsupported(node, what, hint_key=None):
    hint = _HINTS.get(hint_key or "", "") or ""
    raise UnsupportedOpError(f"unsupported op '{what}' at fx node '{node.name}' ({node.op} {node.target}). " + hint +
                             " Supported: Linear, ReLU, Conv2d (groups=1, dilation=1), AvgPool2d/AdaptiveAvgPool2d, Flatten/view/reshape, "
                             "Add (residual), Embedding on the input, Identity/Dropout(eval), argmax at the end.")


@dataclass
class U:                      # materialized non-negative activation
    tid: str
    shape: tuple
    phys: np.ndarray = None   # phys[i] = logical flat index stored at physical position i (None = identity)
    posneg: bool = False      # input encoded as [relu(x); relu(-x)]
    n: int = 0                # logical element count


@dataclass
class Term:
    tid: str
    M: np.ndarray             # dense: (len(tid), N)   grouped: (K, Co) shared over P positions
    P: int = 0                # 0 = dense


@dataclass
class Aff:
    terms: list
    bias: np.ndarray          # dense: (N,)   grouped: (Co,)
    shape: tuple
    phys: np.ndarray          # physical -> logical map of the OUTPUT (None = identity)
    nonneg: bool = False
    grouped: bool = False


@dataclass
class _Batch: pass


_BATCH = _Batch()


@dataclass
class Idx:
    shape: tuple


def _phys(v, n):
    return np.arange(n) if v.phys is None else v.phys


def _prod(s): return int(np.prod(s)) if len(s) else 1


class _Builder:
    def __init__(self, input_shape, input_kind, posneg):
        self.nodes, self.cnt, self.input_shape, self.input_kind, self.posneg = [], {}, tuple(input_shape), input_kind, posneg
        self.argmax = False

    def name(self, base):
        self.cnt[base] = self.cnt.get(base, -1) + 1
        return f"{base}{self.cnt[base]}"

    def emit(self, n):
        self.nodes.append(n); return n

    def length(self, tid):
        return next(n.out_len for n in self.nodes if n.name == tid)

    # ---- value constructors ----
    def make_input(self):
        n = _prod(self.input_shape)
        if self.input_kind == "index": return Idx(self.input_shape)
        if self.posneg:
            self.emit(Node("x", "input", [], 2 * n, encode="posneg"))
            return U("x", self.input_shape, None, True, n)
        self.emit(Node("x", "input", [], n, encode="uint"))
        return U("x", self.input_shape, None, False, n)

    # ---- affine pieces ----
    def dense_term(self, u, M_logical):
        """Term reading U `u` with weight matrix indexed by LOGICAL input position (n_logical, N)."""
        n = u.n
        Mp = M_logical[_phys(u, n)]                    # rows in physical order
        if u.posneg: Mp = np.vstack([Mp, -Mp])         # x = pos - neg
        return Term(u.tid, Mp)

    def linear(self, node, v, mod):
        W = mod.weight.detach().double().numpy().T       # (in, out)
        b = np.zeros(W.shape[1]) if mod.bias is None else mod.bias.detach().double().numpy()
        if isinstance(v, Aff):
            if v.grouped: _unsupported(node, "Linear on an un-activated Conv2d output (put a ReLU or AvgPool between)")
            if len(v.shape) != 1 or _prod(v.shape) != W.shape[0]: _unsupported(node, f"Linear on shape {v.shape} (flatten to 1-D first)")
            Wp = W[_phys(v, W.shape[0])]
            return Aff([Term(t.tid, t.M @ Wp) for t in v.terms], v.bias @ Wp + b, (W.shape[1],), None, False)
        if v.n != W.shape[0] or len(v.shape) != 1:
            _unsupported(node, f"Linear on shape {v.shape} (only a flat feature vector; Flatten first)")
        return Aff([self.dense_term(v, W)], b, (W.shape[1],), None, False)

    def conv(self, node, v, mod):
        if isinstance(v, Aff):
            if not v.nonneg: _unsupported(node, "Conv2d on an un-activated tensor (put a ReLU before it)")
            v = self.materialize(v, "relu")
        if mod.groups != 1 or mod.dilation != (1, 1) or mod.padding_mode != "zeros" or isinstance(mod.padding, str):
            _unsupported(node, f"Conv2d(groups={mod.groups}, dilation={mod.dilation}, padding={mod.padding})")
        C, H, Wd = v.shape; kh, kw = mod.kernel_size; sh, sw = mod.stride; ph, pw = mod.padding
        Co = mod.out_channels; assert C == mod.in_channels
        Ho, Wo = (H + 2 * ph - kh) // sh + 1, (Wd + 2 * pw - kw) // sw + 1; P, K = Ho * Wo, C * kh * kw
        pos = np.empty(v.n, dtype=np.int64); pos[_phys(v, v.n)] = np.arange(v.n)       # logical -> physical
        oh, ow = np.meshgrid(np.arange(Ho), np.arange(Wo), indexing="ij")
        ci, ki, kj = np.meshgrid(np.arange(C), np.arange(kh), np.arange(kw), indexing="ij")
        r = oh.reshape(-1, 1) * sh - ph + ki.reshape(1, -1); c = ow.reshape(-1, 1) * sw - pw + kj.reshape(1, -1)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < Wd)
        lg = ci.reshape(1, -1) * H * Wd + np.clip(r, 0, H - 1) * Wd + np.clip(c, 0, Wd - 1)
        idx = np.where(ok, pos[lg], -1)                                                  # (P, K) physical positions
        Wc = mod.weight.detach().double().numpy().reshape(Co, K).T                      # (K, Co)
        if v.posneg:
            idx = np.concatenate([idx, np.where(idx >= 0, idx + v.n, -1)], 1); Wc = np.vstack([Wc, -Wc])
        g = self.emit(Node(self.name("im2col"), "gather", [v.tid], idx.size, index=idx.reshape(-1).astype(np.int64)))
        b = np.zeros(Co) if mod.bias is None else mod.bias.detach().double().numpy()
        phys = (np.arange(Co)[None, :] * P + np.arange(P)[:, None]).reshape(-1)         # (p, co) -> logical co*P + p
        return Aff([Term(g.name, Wc, P)], b, (Co, Ho, Wo), phys, False, True)

    def pool(self, node, v, k, stride):
        if isinstance(v, Aff): _unsupported(node, "AvgPool on an un-activated tensor (put a ReLU before it)")
        C, H, Wd = v.shape; kh, kw = k; sh, sw = stride; Ho, Wo = (H - kh) // sh + 1, (Wd - kw) // sw + 1
        M = np.zeros((C * H * Wd, C * Ho * Wo))
        for c in range(C):
            for oh in range(Ho):
                for ow in range(Wo):
                    o = c * Ho * Wo + oh * Wo + ow
                    for i in range(kh):
                        for j in range(kw): M[c * H * Wd + (oh * sh + i) * Wd + ow * sw + j, o] = 1.0 / (kh * kw)
        return Aff([self.dense_term(v, M)], np.zeros(M.shape[1]), (C, Ho, Wo), None, True)

    def embedding(self, node, v, mod):
        if not isinstance(v, Idx): _unsupported(node, "Embedding on a computed tensor (only on the network's integer input)")
        T = _prod(v.shape); V, E = mod.weight.shape
        self.emit(Node("x", "input", [], T * V, encode="onehot", vocab=V))
        Wt = mod.weight.detach().double().numpy()
        M = np.zeros((T * V, T * E))
        for t in range(T): M[t * V:(t + 1) * V, t * E:(t + 1) * E] = Wt
        return Aff([Term("x", M)], np.zeros(T * E), tuple(v.shape) + (E,), None, False)

    def as_terms(self, v, out_phys, node):
        """Express v (U or Aff) as terms/bias whose OUTPUT columns follow out_phys (physical->logical)."""
        if isinstance(v, U):
            n = v.n; pu = _phys(v, n)
            M = np.zeros((n, n)); lp = np.empty(n, dtype=np.int64); lp[out_phys] = np.arange(n)   # logical -> out column
            M[np.arange(n), lp[pu]] = 1.0
            return [Term(v.tid, np.vstack([M, -M]) if v.posneg else M)], np.zeros(n), False
        same = (v.phys is None and out_phys is None) or (v.phys is not None and out_phys is not None and np.array_equal(v.phys, out_phys))
        n = len(out_phys)
        if same: return v.terms, v.bias, v.grouped
        if v.grouped: _unsupported(node, "Add of conv outputs with different memory layouts")
        pv = _phys(v, n); lp = np.empty(n, dtype=np.int64); lp[pv] = np.arange(n)
        col = lp[out_phys]                                # out column p reads v column col[p]
        return [Term(t.tid, t.M[:, col]) for t in v.terms], v.bias[col], False

    def add(self, node, a, b):
        if not all(isinstance(z, (U, Aff)) for z in (a, b)): _unsupported(node, "add with a constant/non-tensor operand")
        if a.shape != b.shape: _unsupported(node, f"add of shapes {a.shape} and {b.shape} (no broadcasting)")
        n = _prod(a.shape); op = _phys(a, n)                       # output layout = a's layout
        ta, ba, ga = self.as_terms(a, op, node); tb, bb, gb = self.as_terms(b, op, node)
        if ga or gb: _unsupported(node, "residual Add involving a Conv2d output (use residuals on flat features)")
        nonneg = all(isinstance(z, U) or z.nonneg for z in (a, b))
        return Aff(list(ta) + list(tb), ba + bb, a.shape, a.phys, nonneg, False)

    # ---- materialize ----
    def materialize(self, v, act):
        """Aff -> U via one fused matvec node (act: relu | none)."""
        if isinstance(v, U): return v
        if v.grouped:
            t = v.terms[0]; assert len(v.terms) == 1
            nd = self.emit(Node(self.name("mv"), "matvec", [t.tid], t.P * t.M.shape[1], W=t.M, b=v.bias, act=act, groups=t.P))
        else:
            merged, order = {}, []
            for t in v.terms:
                if t.tid in merged: merged[t.tid] = merged[t.tid] + t.M
                else: merged[t.tid] = t.M.copy(); order.append(t.tid)
            if len(order) == 1: src = order[0]
            else: src = self.emit(Node(self.name("cat"), "concat", order, sum(self.length(t) for t in order))).name
            W = np.vstack([merged[t] for t in order])
            nd = self.emit(Node(self.name("mv"), "matvec", [src], W.shape[1], W=W, b=v.bias, act=act, groups=1))
        return U(nd.name, v.shape, v.phys, False, _prod(v.shape))


def _get_mod(gm, node): return gm.get_submodule(node.target)


def from_torch(model: nn.Module, input_shape, input_kind="float", input_posneg=False) -> Graph:
    """Trace `model` with torch.fx and build the float IR graph.  input_shape excludes the batch dim.
    input_kind="float": non-negative float features (quantized to uint by quant.quantize).  input_posneg=True accepts signed
    inputs by feeding [relu(x); relu(-x)] with weights [W; -W] (doubles the input width).
    input_kind="index": integer token ids (shape (T,)); the first op must be nn.Embedding."""
    model = model.eval()
    try:
        gm = fx.symbolic_trace(model)
    except Exception as e:
        raise UnsupportedOpError(f"torch.fx could not trace the model (data-dependent control flow is not supported): {e}") from e
    B = _Builder(input_shape, input_kind, input_posneg); env = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            env[node.name] = B.make_input(); continue
        if node.op == "output":
            out = env[node.args[0].name] if isinstance(node.args[0], fx.Node) else _unsupported(node, "non-tensor output")
            break
        args = [env[a.name] if isinstance(a, fx.Node) else a for a in node.args]
        if node.op == "call_function" and node.target is operator.getitem and isinstance(args[0], tuple) and args[0][0] == "size":
            env[node.name] = _BATCH if args[1] == 0 else args[0][1][args[1] - 1]; continue
        v = args[0] if args else None
        if node.op == "call_module":
            m = _get_mod(gm, node); t = type(m)
            if t is nn.Linear: r = B.linear(node, v, m)
            elif t is nn.Conv2d: r = B.conv(node, v, m)
            elif t is nn.ReLU: r = _relu(B, node, v)
            elif t in (nn.Identity, nn.Dropout): r = v
            elif t is nn.Flatten:
                if m.start_dim != 1 or m.end_dim != -1: _unsupported(node, f"Flatten({m.start_dim},{m.end_dim})")
                r = _reshape(v, (_prod(v.shape),))
            elif t is nn.AvgPool2d:
                k = (m.kernel_size,) * 2 if isinstance(m.kernel_size, int) else tuple(m.kernel_size)
                s = m.stride if m.stride is not None else m.kernel_size; s = (s,) * 2 if isinstance(s, int) else tuple(s)
                if m.padding not in (0, (0, 0)): _unsupported(node, "AvgPool2d with padding")
                r = B.pool(node, v, k, s)
            elif t is nn.AdaptiveAvgPool2d:
                o = (m.output_size,) * 2 if isinstance(m.output_size, int) else tuple(m.output_size)
                H, W_ = v.shape[1:]
                if H % o[0] or W_ % o[1]: _unsupported(node, "AdaptiveAvgPool2d with non-divisible size")
                k = (H // o[0], W_ // o[1]); r = B.pool(node, v, k, k)
            elif t is nn.Embedding: r = B.embedding(node, v, m)
            else:
                key = {nn.Softmax: "softmax", nn.LogSoftmax: "log_softmax", nn.LayerNorm: "layer_norm", nn.GELU: "gelu",
                       nn.SiLU: "silu", nn.Sigmoid: "sigmoid", nn.Tanh: "tanh", nn.MaxPool2d: "max_pool",
                       nn.MultiheadAttention: "multi_head_attention", nn.BatchNorm1d: "batch_norm", nn.BatchNorm2d: "batch_norm"}.get(t)
                _unsupported(node, t.__name__, key)
        elif node.op == "call_function":
            f = node.target
            if f in (operator.add, torch.add): r = B.add(node, args[0], args[1])
            elif f in (F.relu, torch.relu): r = _relu(B, node, v)
            elif f is torch.flatten:
                r = _reshape(v, (_prod(v.shape),)) if (len(args) < 2 or args[1] == 1) else _unsupported(node, "flatten(start_dim!=1)")
            elif f is F.avg_pool2d:
                k = args[1] if isinstance(args[1], tuple) else (args[1],) * 2
                st = node.kwargs.get("stride", None) or (args[2] if len(args) > 2 else None) or k
                st = st if isinstance(st, tuple) else (st,) * 2; r = B.pool(node, v, k, st)
            elif f is torch.argmax:
                B.argmax = True; r = v
            elif f in (F.softmax, torch.softmax): _unsupported(node, "softmax", "softmax")
            elif f in (F.log_softmax, torch.log_softmax): _unsupported(node, "log_softmax", "log_softmax")
            elif f is F.layer_norm: _unsupported(node, "layer_norm", "layer_norm")
            elif f is F.gelu: _unsupported(node, "gelu", "gelu")
            elif f in (F.silu,): _unsupported(node, "silu", "silu")
            elif f in (torch.sigmoid, F.sigmoid): _unsupported(node, "sigmoid", "sigmoid")
            elif f in (torch.tanh, F.tanh): _unsupported(node, "tanh", "tanh")
            elif f in (operator.mul, torch.mul): _unsupported(node, "mul", "mul")
            elif f in (torch.matmul, operator.matmul, torch.bmm): _unsupported(node, "matmul", "matmul")
            elif f is F.max_pool2d: _unsupported(node, "max_pool2d", "max_pool")
            else: _unsupported(node, getattr(f, "__name__", str(f)))
        elif node.op == "call_method":
            m = node.target
            if m == "relu": r = _relu(B, node, v)
            elif m in ("view", "reshape"):
                shp = list(args[1]) if len(args) == 2 and isinstance(args[1], (tuple, list)) else list(args[1:])
                if shp and (shp[0] is _BATCH or (len(shp) > 1 and shp[0] in (-1, 1))): shp = shp[1:]   # drop the batch dim
                if any(not isinstance(z, int) for z in shp): _unsupported(node, "view/reshape with a dynamic size")
                if -1 in shp:
                    rest = _prod([z for z in shp if z != -1]); shp = [_prod(v.shape) // rest if z == -1 else z for z in shp]
                if _prod(shp) != _prod(v.shape): _unsupported(node, f"reshape {v.shape} -> {tuple(shp)} changes the element count (batch dim?)")
                r = _reshape(v, tuple(shp))
            elif m == "size":
                r = ("size", v.shape) if len(args) == 1 else (_BATCH if args[1] == 0 else v.shape[args[1] - 1])
            elif m == "flatten": r = _reshape(v, (_prod(v.shape),))
            elif m == "add": r = B.add(node, args[0], args[1])
            elif m == "argmax": B.argmax = True; r = v
            elif m == "softmax": _unsupported(node, "softmax", "softmax")
            elif m in ("sigmoid", "tanh"): _unsupported(node, m, m)
            elif m in ("mul", "matmul"): _unsupported(node, m, m)
            else: _unsupported(node, m)
        else:
            _unsupported(node, f"{node.op}:{node.target}")
        env[node.name] = r
    out = B.materialize(out, "none")
    g = Graph(B.nodes, out.tid, argmax=B.argmax, profile="float",
              meta=dict(input_shape=tuple(input_shape), output_phys=None if out.phys is None else out.phys.tolist(),
                        output_shape=tuple(out.shape)))
    if out.phys is not None and not np.array_equal(out.phys, np.arange(len(out.phys))):
        # network ends in conv-layout (P, C) output: documented in meta so callers can restore torch order
        g.meta["output_layout"] = "phys"
    return g.validate()


def _reshape(v, shape):
    if isinstance(v, U): return U(v.tid, shape, v.phys, v.posneg, v.n)
    if isinstance(v, Aff): return Aff(v.terms, v.bias, shape, v.phys, v.nonneg, v.grouped)
    return Idx(shape)


def _relu(B, node, v):
    if isinstance(v, U): return v                      # activations are already >= 0
    if isinstance(v, Idx): _unsupported(node, "relu on integer ids")
    return B.materialize(v, "relu")
