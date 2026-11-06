"""Paged execution ("weight streaming"): a network larger than one machine runs as a sequence of pages on ONE shared Spec.

A page = the output neurons of one matvec node that fit the column / op budget, with exactly the source registers they read.  Each page is a
normal machine program (sched.compile_neurons) run from a fresh prompt (fresh DeltaNet state).  The host carries the integer activations
(0..amax) between pages in an activation store; inside a page the model is driven by its own tokens (argmax; append).  See docs/STREAMING.md.
"""
from dataclasses import dataclass, field

import numpy as np

from ..ir import Graph, Node
from .circuit import build
from .sched import Compiled, DoesNotFit, Neuron, compile_neurons, wide_needed
from .spec import Spec, group_fan, GROUP_F


class PageMismatch(AssertionError):
    """A page's circuit output differs from the token golden (an executor bug)."""


@dataclass
class Page:
    node: str
    neurons: list          # [{name, wts: {element key: int}, bias, shift, key}]  (host-side description)
    comp: Compiled = None
    src: list = field(default_factory=list)     # element keys feeding comp.in_regs (same order)
    dst: list = field(default_factory=list)     # element keys produced, in comp.out_regs order


@dataclass
class PagedCompiled:
    spec: Spec
    graph: Graph
    rows: int
    pages: list
    in_keys: list
    out_keys: list         # element key per output element, None = constant 0

    def run(self, xq, execute):
        """One ENCODED input vector -> (output vector, [(prompt tokens, emitted tokens) per page]).
        execute(prog) -> the emitted tokens of one page (the model's argmax chain, or a golden)."""
        xq = np.asarray(xq, dtype=np.int64).reshape(-1)
        assert len(xq) == len(self.in_keys)
        store = {k: int(v) for k, v in zip(self.in_keys, xq)}
        sizes = []
        for pg in self.pages:
            prog = pg.comp.program(np.array([store[k] for k in pg.src], dtype=np.int64))
            em = execute(prog)
            sizes.append((len(prog.prompt()), len(em)))
            for k, v in zip(pg.dst, pg.comp.decode(em)): store[k] = int(v)
        return np.array([store[k] if k is not None else 0 for k in self.out_keys], dtype=np.int64), sizes

    def stats(self):
        sp = self.spec
        return dict(pages=len(self.pages), neurons=sum(len(p.neurons) for p in self.pages), spec=str(sp), columns=sp.ncols, tiles=sp.tiles,
                    page_neurons=[len(p.neurons) for p in self.pages], page_columns=[p.comp.spec.ncols for p in self.pages],
                    page_cells=[len(p.comp.W) for p in self.pages], nvocab=sp.nvocab, layers=build(sp)[0].n_layers())


# ---------------------------------------------------------------------------------------------------------------- executors
def golden_executor():
    """The numpy token golden of a page (asm.reference_emit): fast, no circuit."""
    return lambda prog: prog.expected()


def sim_executor(spec):
    """The CPU circuit-IR simulator, checked against the token golden for every page."""
    from .run import IRMachine
    m = IRMachine(spec)

    def ex(prog):
        p = prog.prompt(); em = prog.expected(); pred = m.run(p, em)
        if pred != em + [spec.HALT]: raise PageMismatch("circuit simulator disagrees with the token golden on a page")
        return em
    return ex


def model_executor(spec, feed_factory):
    """The real model: feed_factory() -> a fresh feed(tokens) (fresh state per page); host = argmax; append."""
    from .run import generate
    return lambda prog: generate(feed_factory(), prog.prompt(), spec.HALT)


# ---------------------------------------------------------------------------------------------------------------- compile
def _elements(g):
    """Walk the graph: element key lists per node, and the neuron descriptions of every matvec node."""
    vals, per_node, in_keys = {}, [], []
    for node in g.nodes:
        if node.op == "input":
            vals[node.name] = [(node.name, i) for i in range(node.out_len)]; in_keys = vals[node.name]
        elif node.op == "gather":
            src = vals[node.inputs[0]]; vals[node.name] = [src[i] if i >= 0 else None for i in node.index]
        elif node.op == "concat":
            vals[node.name] = [k for i in node.inputs for k in vals[i]]
        else:
            x = vals[node.inputs[0]]; K, N = node.W.shape; out, nus = [], []
            for p in range(node.groups):
                xs = x[p * K:(p + 1) * K]
                for n in range(N):
                    wts = {}
                    for k in range(K):
                        w = int(node.W[k, n])
                        if w and xs[k] is not None: wts[xs[k]] = wts.get(xs[k], 0) + w
                    key = (node.name, p * N + n); out.append(key)
                    nus.append(dict(name=f"{node.name}[{p},{n}]", wts={k: w for k, w in wts.items() if w}, bias=int(node.b[n]), shift=node.shift, key=key))
            vals[node.name] = out; per_node.append((node.name, nus))
    return per_node, in_keys, vals[g.output]


def _compile_page(g, rows, neurons, spec_override=None, wide=None):
    src, seen = [], {}
    for nu in neurons:
        for k in nu["wts"]:
            if k not in seen: seen[k] = len(src); src.append(k)
    dr = rows - g.bias_rows
    reg = lambda k: (k // dr) * rows + k % dr
    in_regs = [reg(i) for i in range(len(src))]
    ns = [Neuron(nu["name"], {in_regs[seen[k]]: w for k, w in nu["wts"].items()}, nu["bias"], nu["shift"], reg(len(src) + i))
          for i, nu in enumerate(neurons)]
    hdr = Graph([Node("x", "input", [], max(1, len(src)))], "x", ab=g.ab, wb=g.wb, profile=g.profile, bias_rows=g.bias_rows)
    comp = compile_neurons(hdr, rows, ns, in_regs, [n.dest for n in ns], len(src) + len(ns), spec_override, wide=wide)
    return comp, src


def fan_cap(g, rows, smax):
    """Largest `fan` (columns summed by ONE adder tree) that builds at this row count, bit widths and max shift AND stays exact: fan * rows is
    capped at spec.GROUP_F (a bigger single tree can build but its shift mux produces bf16 values it cannot represent, docs/WIDE.md section 5)."""
    cap = 0
    for f in range(1, group_fan(rows) + 1):
        try: build(Spec(rows=rows, cols=(1,) * f, ab=g.ab, wb=g.wb, nops=1, fan=f, smax=smax))
        except (ValueError, AssertionError): break
        cap = f
    return cap


def _pack(g, rows, per_node, max_cols, max_ops, fcap, smax_g, tile_caps=None, wide=None):
    """Greedy pages per matvec node: the largest prefix of the remaining neurons that fits (monotone in the prefix, found by bisection)."""
    pages = []
    for name, nus in per_node:
        i = 0
        while i < len(nus):
            def fits(n):
                try: comp, _ = _compile_page(g, rows, nus[i:i + n], wide=wide)
                except DoesNotFit: return False
                sp = comp.spec
                if sp.ncols > max_cols or sp.fan > fcap: return False
                return tile_caps is None or all(c <= (tile_caps[t] if t < len(tile_caps) else 0) for t, c in enumerate(sp.cols))
            if not fits(1):
                comp, _ = _compile_page(g, rows, nus[i:i + 1], wide=wide)
                if comp.spec.fan > fcap:
                    raise DoesNotFit(f"{nus[i]['name']}: the neuron reads up to {comp.spec.fan * (rows - g.bias_rows)} registers (fan {comp.spec.fan} at {rows} rows); the circuit "
                                     f"is exact up to fan {fcap} per adder tree at {rows} rows (fan*rows <= {GROUP_F}, docs/WIDE.md)")
                raise DoesNotFit(f"{nus[i]['name']}: a single neuron needs columns {comp.spec.cols} (fan {comp.spec.fan}); max_cols={max_cols}, per-tile caps {tile_caps}")
            lo, hi = 1, min(max_ops, len(nus) - i)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if fits(mid): lo = mid
                else: hi = mid - 1
            pages.append(Page(name, nus[i:i + lo])); i += lo
    return pages


def _shared_spec(g, rows, pages, wide=None):
    """One machine for all pages: elementwise maximum of the pages' needs."""
    nat = [_compile_page(g, rows, p.neurons, wide=wide)[0].spec for p in pages]
    tiles = max(s.tiles for s in nat)
    cols = tuple(max([s.cols[t] if t < s.tiles else 0 for s in nat]) for t in range(tiles))
    return Spec(rows=rows, cols=cols, ab=g.ab, wb=g.wb, nops=max(s.nops for s in nat), fan=max(s.fan for s in nat), smax=max(s.smax for s in nat),
                tpg=max(s.tpg for s in nat), gfan=max(s.gfan for s in nat))


def compile_paged(g, rows=16, max_cols=60, max_ops=60):
    """Cut `g` (an integer machine Graph) into pages of at most `max_cols` physical columns and `max_ops` ops; return PagedCompiled.
    The shared machine has the elementwise maximum of the pages' columns per tile, which can exceed `max_cols` when different pages load
    different tiles; a second packing pass then caps every tile's columns proportionally.
    DoesNotFit names the limit that failed (a single neuron too wide for the circuit, columns, ops, the 126-column mask head)."""
    g.validate(machine=True)
    per_node, in_keys, out_keys = _elements(g)
    max_ops = min(max_ops, 128)
    smax_g = max([n.shift for n in g.matvecs()] + [0])
    wide = wide_needed([Neuron(nu["name"], nu["wts"], nu["bias"], nu["shift"], 0) for _, nus in per_node for nu in nus], rows, g.bias_rows)
    # a neuron wider than one adder tree: tile groups (docs/WIDE.md); the number of columns one op may read is then bounded by the mask width only
    fcap = min(max_cols, 126) if wide else fan_cap(g, rows, smax_g)
    try:
        pages = _pack(g, rows, per_node, max_cols, max_ops, fcap, smax_g, wide=wide)
    except DoesNotFit as e:
        if wide or "reads up to" not in str(e): raise
        wide = True; fcap = min(max_cols, 126)              # split weight columns pushed a neuron past one tree: use tile groups
        pages = _pack(g, rows, per_node, max_cols, max_ops, fcap, smax_g, wide=wide)
    spec = _shared_spec(g, rows, pages, wide=wide)
    if spec.ncols > max_cols and spec.tiles > 1:
        caps = tuple(max(1, c * max_cols // spec.ncols) if c else 0 for c in spec.cols)
        try:
            pages2 = _pack(g, rows, per_node, max_cols, max_ops, fcap, smax_g, tile_caps=caps, wide=wide)
            spec2 = _shared_spec(g, rows, pages2, wide=wide)
            if spec2.ncols < spec.ncols: pages, spec = pages2, spec2
        except DoesNotFit:
            pass
    if spec.ncols > 126:
        raise DoesNotFit(f"the shared machine needs {spec.ncols} physical columns ({spec.cols}); a program-memory mask head holds at most 126. Lower max_cols.")
    try: build(spec)
    except (ValueError, AssertionError) as e:
        raise DoesNotFit(f"a neuron reads up to {spec.fan * (rows - g.bias_rows)} registers (fan {spec.fan} at {rows} rows, smax {spec.smax}); the requantizer's "
                         f"carry staircase must fit its 256 window (docs/WIDE.md): {str(e)[:120]}") from None
    for p in pages:
        p.comp, p.src = _compile_page(g, rows, p.neurons, spec_override=spec, wide=wide)
        p.dst = [nu["key"] for nu in p.neurons]
    return PagedCompiled(spec, g, rows, pages, in_keys, out_keys)
