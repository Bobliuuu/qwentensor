"""Host-side scheduler: qwentensor.ir.Graph (integer IR) -> machine Spec + Program (prompt, expected tokens, output map).

A tensor is a list of register references (-1 = constant 0), so gather / concat / flatten cost nothing: they re-index the
consumer's weights.  A matvec output neuron becomes one OP: the set of physical columns (one or more per touched tile; weights
beyond the 5-bit range are split over several columns) whose accumulator counts are added in the adder tree, its shift, and a
fresh destination register (SSA: a register's successor in the program memory is a property of the register).

Capacity (all explicit, see docs/EXECUTOR.md): registers = inputs + all neurons + bias rows per tile; columns = per neuron the
number of touched tiles; ops <= 128.
"""
from dataclasses import dataclass, field

import numpy as np

from .asm import Op, Program
from .spec import Spec


class DoesNotFit(Exception):
    pass


@dataclass
class Neuron:
    name: str
    wts: dict            # physical register -> int weight (may exceed the 5-bit range: split over columns)
    bias: int
    shift: int
    dest: int            # physical register
    cols: list = field(default_factory=list)


@dataclass
class Compiled:
    spec: Spec
    graph: object
    neurons: list
    W: dict
    ops: dict
    order: list
    in_regs: list
    const_regs: list
    out_regs: list
    entry: int

    def program(self, xq):
        """Program for one ENCODED integer input vector (host: layout + register init only)."""
        xq = np.asarray(xq, dtype=np.int64).reshape(-1)
        assert len(xq) == len(self.in_regs) and 0 <= xq.min() and xq.max() <= self.spec.amax
        init = {r: 1 for r in self.const_regs}
        init.update({r: int(v) for r, v in zip(self.in_regs, xq) if v})
        return Program(self.spec, dict(self.W), dict(self.ops), list(self.order), init, self.entry)

    def decode(self, emitted):
        """Output vector from the model's emitted tokens (detokenisation)."""
        sp = self.spec; val = {}
        for t in emitted:
            if sp.is_act(t): val[sp.act_reg(t)] = sp.act_val(t)
        return np.array([val.get(r, 0) if r >= 0 else 0 for r in self.out_regs], dtype=np.int64)


def _split(w, lo, hi):
    parts = []
    while w:
        p = max(lo, min(hi, w)); parts.append(p); w -= p
    return parts


def compile_graph(g, rows=16, spec_override=None, nops=None, smax=None, fan=None, min_cols=None):
    """Build the Spec (sized for this graph unless `spec_override` is given) and the program skeleton."""
    g.validate(machine=True)
    br = g.bias_rows; dr = rows - br; assert dr >= 1
    lo, hi = g.wlo, g.whi
    counter = [0]

    def new_reg():
        n = counter[0]; counter[0] += 1
        return (n // dr) * rows + n % dr

    vals, neurons, in_regs = {}, [], []
    for node in g.nodes:
        if node.op == "input":
            vals[node.name] = [new_reg() for _ in range(node.out_len)]; in_regs = vals[node.name]
        elif node.op == "gather":
            src = vals[node.inputs[0]]; vals[node.name] = [src[i] if i >= 0 else -1 for i in node.index]
        elif node.op == "concat":
            vals[node.name] = [r for i in node.inputs for r in vals[i]]
        else:
            x = vals[node.inputs[0]]; K, N = node.W.shape; out = []
            for p in range(node.groups):
                xs = x[p * K:(p + 1) * K]
                for n in range(N):
                    wts = {}
                    for k in range(K):
                        w = int(node.W[k, n])
                        if w and xs[k] >= 0: wts[xs[k]] = wts.get(xs[k], 0) + w
                    wts = {r: w for r, w in wts.items() if w}
                    d = new_reg(); out.append(d)
                    neurons.append(Neuron(f"{node.name}[{p},{n}]", wts, int(node.b[n]), node.shift, d))
            vals[node.name] = out
    out_regs = vals[g.output]
    ntile = max(1, -(-counter[0] // dr))
    const_regs = [t * rows + dr + i for t in range(ntile) for i in range(br)]
    # ---- columns: per neuron, one or more columns in every tile it touches; weights beyond [wlo, whi] are split over columns
    used = [0] * ntile; W = {}
    for nu in neurons:
        by_tile = {}
        for r, w in nu.wts.items(): by_tile.setdefault(r // rows, {})[r % rows] = _split(w, lo, hi)
        bparts = _split(nu.bias, lo, hi)
        if len(bparts) > br: raise DoesNotFit(f"{nu.name}: bias {nu.bias} needs {len(bparts)} constant rows, bias_rows={br}")
        btile = min(by_tile) if by_tile else 0
        if bparts: by_tile.setdefault(btile, {})
        for t, rw in sorted(by_tile.items()):
            k = max([len(v) for v in rw.values()] + [1])
            for j in range(k):
                nu.cols.append((t, used[t] + j))
                for s_, parts in rw.items():
                    if j < len(parts): W[(s_, (t, used[t] + j))] = parts[j]
            if bparts and t == btile:
                for i, bp in enumerate(bparts): W[(dr + i, (t, used[t]))] = bp
            used[t] += k
    cols = tuple(used)
    fan_need = max([len(nu.cols) for nu in neurons] + [1])
    smax_need = max([nu.shift for nu in neurons] + [0])
    spec = Spec(rows=rows, cols=cols if min_cols is None else tuple(max(a, b) for a, b in zip(cols, min_cols)),
                ab=g.ab, wb=g.wb, nops=max(nops or 0, len(neurons), 1), fan=max(fan or 0, fan_need),
                smax=max(smax or 0, smax_need)) if spec_override is None else spec_override
    if spec_override is not None:
        sp = spec_override
        if sp.rows != rows or sp.tiles < ntile or any(u > c for u, c in zip(used, sp.cols)) or len(neurons) > sp.nops \
                or fan_need > sp.fan or smax_need > sp.smax or (sp.ab, sp.wb) != (g.ab, g.wb):
            raise DoesNotFit(f"needs tiles={ntile} cols={tuple(used)} fan={fan_need} ops={len(neurons)} smax={smax_need}; "
                             f"machine has {sp}")
    if len(neurons) > 128: raise DoesNotFit(f"{len(neurons)} neurons > 128 op slots (one mask/shift head per 128 ops)")
    sp = spec
    # physical column ids
    cid = lambda tc: sp.col_id(*tc)
    Wp = {(s, cid(tc)): w for (s, tc), w in W.items()}
    ops = {}
    for j, nu in enumerate(neurons): ops[j] = Op([cid(tc) for tc in nu.cols], nu.shift, nu.dest, nu.name)
    entry = const_regs[0]
    return Compiled(sp, g, neurons, Wp, ops, list(range(len(neurons))), in_regs, const_regs, out_regs, entry)
