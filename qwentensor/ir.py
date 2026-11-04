"""qwentensor integer graph IR: the contract between the frontend/quantizer and the machine executor.

Spec and rationale: docs/IR.md.  Every tensor is a FLAT 1-D vector (batch dim handled outside the graph).
Ops: input, gather, concat, matvec.  Integer graphs (profile "v2.0" / "v2.1") are what the machine runs and what
`reference.run` executes bit-exactly; the same container with profile "float" holds the un-quantized graph.
"""
from dataclasses import dataclass, field
import json

import numpy as np

OPS = ("input", "gather", "concat", "matvec")
INT_ACTS = ("relu_clamp", "linear")      # linear = signed raw readout, NOT machine-native (see docs/IR.md)
FLOAT_ACTS = ("relu", "none")
PROFILES = ("v2.0", "v2.1", "float")
ENCODES = ("uint", "onehot", "posneg")


@dataclass
class Node:
    name: str
    op: str
    inputs: list = field(default_factory=list)
    out_len: int = 0
    # matvec
    W: np.ndarray = None        # (K, N) ints in [wlo, whi]   (float profile: float64)
    b: np.ndarray = None        # (N,)  ints in the bias range
    shift: int = 0              # floor right shift of the accumulator, 0..15 (profile v2.0: must be 0)
    act: str = "relu_clamp"     # int: relu_clamp | linear ; float: relu | none
    groups: int = 1             # P: input is P consecutive K-vectors, the same W is applied to each, output is P N-vectors
    # gather
    index: np.ndarray = None    # (out_len,) ints, index into the single input; -1 means constant 0
    # input
    encode: str = "uint"        # uint | onehot | posneg  (how the HOST builds the vector from the raw sample; see docs/IR.md)
    vocab: int = 0              # onehot: classes per token
    # real value of one integer step, per element (float64 array len out_len) -- info for dequantization only
    scale: np.ndarray = None
    meta: dict = field(default_factory=dict)


@dataclass
class Graph:
    nodes: list
    output: str
    argmax: bool = False        # host takes argmax (lowest index on ties) of the output vector
    profile: str = "v2.1"
    ab: int = 5                 # activation bits: values 0..amax
    wb: int = 5                 # weight bits: -2^(wb-1)..2^(wb-1)-1
    bias_rows: int = 1          # bias range is bias_rows * [wlo, whi]; the executor spends this many constant-1 rows per tile
    meta: dict = field(default_factory=dict)

    # ---- ranges ----
    @property
    def amax(self): return (1 << self.ab) - 1
    @property
    def wlo(self): return -(1 << (self.wb - 1))
    @property
    def whi(self): return (1 << (self.wb - 1)) - 1
    @property
    def blo(self): return self.wlo * self.bias_rows
    @property
    def bhi(self): return self.whi * self.bias_rows

    def node(self, name):
        for n in self.nodes:
            if n.name == name: return n
        raise KeyError(name)

    @property
    def input(self):
        ins = [n for n in self.nodes if n.op == "input"]
        assert len(ins) == 1, "exactly one input node supported"
        return ins[0]

    def matvecs(self): return [n for n in self.nodes if n.op == "matvec"]

    # ---- validation ----
    def validate(self, machine=False):
        """Structural + range checks.  machine=True additionally rejects what the v2 datapath cannot execute."""
        assert self.profile in PROFILES, self.profile
        seen = {}
        for n in self.nodes:
            assert n.op in OPS, f"{n.name}: unknown op {n.op}"
            assert n.name not in seen, f"duplicate node {n.name}"
            for i in n.inputs: assert i in seen, f"{n.name}: input {i} not defined earlier (nodes must be topologically ordered)"
            ins = [seen[i].out_len for i in n.inputs]
            if n.op == "input":
                assert not n.inputs and n.encode in ENCODES and n.out_len > 0
                if n.encode == "onehot": assert n.vocab > 0 and n.out_len % n.vocab == 0
                if n.encode == "posneg": assert n.out_len % 2 == 0
            elif n.op == "gather":
                assert len(ins) == 1 and len(n.index) == n.out_len and n.index.min() >= -1 and n.index.max() < ins[0], n.name
            elif n.op == "concat":
                assert len(ins) >= 1 and sum(ins) == n.out_len, n.name
            else:
                assert len(ins) == 1, n.name
                K, N = n.W.shape
                assert ins[0] == n.groups * K and n.out_len == n.groups * N and n.b.shape == (N,), \
                    f"{n.name}: shape mismatch in={ins[0]} W={n.W.shape} groups={n.groups} out={n.out_len}"
                if self.profile == "float":
                    assert n.act in FLOAT_ACTS, n.name
                else:
                    assert n.act in INT_ACTS, n.name
                    assert np.issubdtype(n.W.dtype, np.integer) and np.issubdtype(n.b.dtype, np.integer), n.name
                    assert n.W.min() >= self.wlo and n.W.max() <= self.whi, f"{n.name}: weight out of [{self.wlo},{self.whi}]"
                    assert n.b.min() >= self.blo and n.b.max() <= self.bhi, f"{n.name}: bias out of [{self.blo},{self.bhi}]"
                    assert 0 <= n.shift <= 15, n.name
                    if machine or self.profile == "v2.0":
                        assert self.profile != "v2.0" or n.shift == 0, f"{n.name}: shift != 0 not allowed in profile v2.0"
                    if machine:
                        assert n.act == "relu_clamp", f"{n.name}: act=linear is not machine-native"
            seen[n.name] = n
        assert self.output in seen
        return self

    # ---- machine cost model (what the executor must schedule) ----
    def cost(self, kr=16, nc=16):
        """Per matvec node: tile-matvecs needed on a kr x nc array (kr-1-bias_rows usable input rows, nc usable cols).
        total_tile_matvecs counts every (input tile, output tile) pair for every group position.  Exposed so the
        executor can sanity check its schedule; it is not authoritative."""
        rows = kr - self.bias_rows
        out, tot = {}, 0
        for n in self.matvecs():
            K, N = n.W.shape
            kt, nt = -(-K // rows), -(-N // nc)
            out[n.name] = dict(K=K, N=N, groups=n.groups, in_tiles=kt, out_tiles=nt, tile_matvecs=kt * nt * n.groups,
                               distinct_weight_tiles=kt * nt, shift=n.shift, act=n.act)
            tot += kt * nt * n.groups
        out["total_tile_matvecs"] = tot
        return out

    # ---- serialization ----
    def to_dict(self):
        def a(x): return None if x is None else x.tolist()
        return dict(output=self.output, argmax=self.argmax, profile=self.profile, ab=self.ab, wb=self.wb,
                    bias_rows=self.bias_rows, meta=self.meta,
                    nodes=[dict(name=n.name, op=n.op, inputs=n.inputs, out_len=n.out_len, W=a(n.W),
                                W_dtype=None if n.W is None else str(n.W.dtype), b=a(n.b), shift=n.shift, act=n.act,
                                groups=n.groups, index=a(n.index), encode=n.encode, vocab=n.vocab, scale=a(n.scale),
                                meta=n.meta) for n in self.nodes])

    @staticmethod
    def from_dict(d):
        ns = []
        for e in d["nodes"]:
            e = dict(e); dt = e.pop("W_dtype")
            for k in ("W",):
                e[k] = None if e[k] is None else np.array(e[k], dtype=dt)
            e["b"] = None if e["b"] is None else np.array(e["b"], dtype=np.float64 if dt and "float" in dt else np.int64)
            e["index"] = None if e["index"] is None else np.array(e["index"], dtype=np.int64)
            e["scale"] = None if e["scale"] is None else np.array(e["scale"], dtype=np.float64)
            ns.append(Node(**e))
        return Graph(ns, **{k: d[k] for k in ("output", "argmax", "profile", "ab", "wb", "bias_rows", "meta")})

    def save(self, path):
        with open(path, "w") as f: json.dump(self.to_dict(), f)

    @staticmethod
    def load(path):
        with open(path) as f: return Graph.from_dict(json.load(f))

    def summary(self):
        lines = [f"Graph profile={self.profile} ab={self.ab} wb={self.wb} bias_rows={self.bias_rows} argmax={self.argmax} output={self.output}"]
        for n in self.nodes:
            extra = ""
            if n.op == "matvec": extra = f" W{tuple(n.W.shape)} groups={n.groups} shift={n.shift} act={n.act}"
            if n.op == "input": extra = f" encode={n.encode}"
            lines.append(f"  {n.name:12s} {n.op:7s} in={n.inputs} len={n.out_len}{extra}")
        return "\n".join(lines)
