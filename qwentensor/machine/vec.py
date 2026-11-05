"""Vectorized machine: one emitted token carries `lanes` neuron results, and the READ token is fused into the register write.

Design, exactness argument and predicted effect: docs/VECTORIZED.md.  The scalar machine (spec/circuit/asm/sched) is untouched.

Registers.  A *group* g = block * rows + row holds `lanes` registers, one per lane; register (g, l) is flat id g * lanes + l and lives at
row `g mod rows` of tile `(g div rows) * lanes + l`.  The top `bias_rows` rows of every block are constant-1 groups.

Tokens (ids ascending):
  0 HALT   1 NOP
  VEC(g; v_0..v_{L-1})    writes lane l's value into register (g, l) for every l, then queries the accumulators with the mask stored under slot g;
                          its successor token is VEC(succ[g]; the L requantised values)          (the only token the model emits besides HALT)
  ACT(r; v)               prompt-only single register write (never queries)
  CELL(s,c,w)             weight write (prompt only)
  PSG(g, gd)              program memory: succ[g] = VEC(gd; .) (gd == ngroups: HALT)
  PM(l,f,g,cl)            mask field f of lane l under slot g := the cl-th column of lane class l (cl == ncmax: none)
  PSH(l,g,s)              shift of lane l under slot g := s
"""
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS

import numpy as np

from qwentpu.compiler import evaluate as ir, nonlin as N
from qwentpu.compiler.gadgets import Circ
from qwentpu.compiler.ir import A_BIT, A_REF, K, Expr

from .sched import DoesNotFit, _split

S_CODE = 2 ** 12


@dataclass(frozen=True)
class VSpec:
    rows: int = 16
    lanes: int = 2
    cols: tuple = (4, 4)           # physical columns per tile; len = blocks * lanes
    ccls: tuple = ()               # lane class of every column (tile-major order); empty = all class 0
    ab: int = 5
    wb: int = 5
    fan: int = 1
    smax: int = 0

    def __post_init__(self):
        assert len(self.cols) % self.lanes == 0
        if not self.ccls: object.__setattr__(self, "ccls", (0,) * sum(self.cols))
        assert len(self.ccls) == sum(self.cols) and all(0 <= c < self.lanes for c in self.ccls)

    # ---- geometry ----
    @cached_property
    def tiles(self): return len(self.cols)
    @cached_property
    def nblk(self): return self.tiles // self.lanes
    @cached_property
    def ngroups(self): return self.nblk * self.rows
    @cached_property
    def nreg(self): return self.ngroups * self.lanes
    @cached_property
    def ncols(self): return sum(self.cols)
    @cached_property
    def nout(self): return 1 << self.ab
    @cached_property
    def amax(self): return self.nout - 1
    @cached_property
    def nw(self): return 1 << self.wb
    @cached_property
    def wlo(self): return -(1 << (self.wb - 1))
    @cached_property
    def whi(self): return (1 << (self.wb - 1)) - 1
    @cached_property
    def NV(self): return self.nout ** self.lanes
    @cached_property
    def cls_cols(self): return [[c for c in range(self.ncols) if self.ccls[c] == l] for l in range(self.lanes)]
    @cached_property
    def cl_index(self):
        out = {}
        for l, cs in enumerate(self.cls_cols):
            for i, c in enumerate(cs): out[c] = i
        return out
    @cached_property
    def ncmax(self): return max(1, max(len(c) for c in self.cls_cols))

    def col_tile(self, c):
        for t, n in enumerate(self.cols):
            if c < n: return t
            c -= n
        raise IndexError
    def col_base(self, t): return sum(self.cols[:t])
    def col_id(self, t, i):
        assert i < self.cols[t]; return self.col_base(t) + i

    def reg_group(self, r): return r // self.lanes
    def reg_lane(self, r): return r % self.lanes
    def reg_tile(self, r): g = r // self.lanes; return (g // self.rows) * self.lanes + r % self.lanes
    def reg_row(self, r): return (r // self.lanes) % self.rows
    def tile_reg(self, t, s): return ((t // self.lanes) * self.rows + s) * self.lanes + t % self.lanes

    # ---- token layout ----
    HALT = 0
    NOP = 1
    @cached_property
    def VEC0(self): return self.NV
    @cached_property
    def ACT0(self): return self.VEC0 + self.ngroups * self.NV
    @cached_property
    def CELL0(self): return self.ACT0 + self.nreg * self.nout
    @cached_property
    def PSG0(self): return self.CELL0 + self.rows * self.ncols * self.nw
    @cached_property
    def PM0(self): return self.PSG0 + self.ngroups * (self.ngroups + 1)
    @cached_property
    def PSH0(self): return self.PM0 + self.lanes * self.fan * self.ngroups * (self.ncmax + 1)
    @cached_property
    def nvocab(self): return self.PSH0 + (self.lanes * self.ngroups * (self.smax + 1) if self.smax else 0)
    @cached_property
    def code_bits(self): return max(8, (self.nvocab - 1).bit_length())

    def vec(self, g, vs):
        assert len(vs) == self.lanes
        return self.VEC0 + g * self.NV + sum(int(v) * self.nout ** l for l, v in enumerate(vs))
    def act(self, r, v): return self.ACT0 + r * self.nout + int(v)
    def cell(self, s, c, w): return self.CELL0 + (s * self.ncols + c) * self.nw + (int(w) & (self.nw - 1))
    def psg(self, g, gd): return self.PSG0 + g * (self.ngroups + 1) + gd
    def pm(self, l, f, g, cl): return self.PM0 + ((l * self.fan + f) * self.ngroups + g) * (self.ncmax + 1) + cl
    def psh(self, l, g, s): return self.PSH0 + (l * self.ngroups + g) * (self.smax + 1) + s

    def is_vec(self, t): return self.VEC0 <= t < self.ACT0
    def is_act(self, t): return self.ACT0 <= t < self.CELL0
    def is_cell(self, t): return self.CELL0 <= t < self.PSG0
    def vec_dec(self, t):
        g, v = divmod(t - self.VEC0, self.NV)
        return g, [(v // self.nout ** l) % self.nout for l in range(self.lanes)]
    def act_dec(self, t): return divmod(t - self.ACT0, self.nout)
    def cell_dec(self, t):
        u, w = divmod(t - self.CELL0, self.nw); s, c = divmod(u, self.ncols)
        return s, c, (w - self.nw if w >= self.nw // 2 else w)
    def prog_dec(self, t):
        if self.PSG0 <= t < self.PM0:
            g, gd = divmod(t - self.PSG0, self.ngroups + 1); return ("PSG", g, gd)
        if self.PM0 <= t < self.PSH0:
            u, cl = divmod(t - self.PM0, self.ncmax + 1); lf, g = divmod(u, self.ngroups); l, f = divmod(lf, self.fan)
            return ("PM", l, f, g, cl)
        if self.smax and self.PSH0 <= t < self.nvocab:
            u, s = divmod(t - self.PSH0, self.smax + 1); l, g = divmod(u, self.ngroups); return ("PSH", l, g, s)
        return None

    def table(self):
        """Per-token-id numpy decode arrays; every circuit token feature is a function of these (vectorised, no per-token python)."""
        n, L = self.nvocab, self.lanes
        ids = np.arange(n); z = lambda v=-1: np.full(n, v, np.int64)
        kind = np.zeros(n, np.int8); G = z(); R = z(); TL = z(); RT = z(); S = z(); C = z(); Wv = z(); SW = z(0); VALID = z(0)
        V = [z() for _ in range(L)]; PL = z(); PF = z(); PCL = z(); PS = z()
        lo, hi = self.VEC0, self.ACT0; u = ids[lo:hi] - lo; kind[lo:hi] = 1
        G[lo:hi] = u // self.NV; R[lo:hi] = G[lo:hi] % self.rows
        for l in range(L): V[l][lo:hi] = (u % self.NV // self.nout ** l) % self.nout
        lo, hi = self.ACT0, self.CELL0; u = ids[lo:hi] - lo; kind[lo:hi] = 2; r = u // self.nout
        G[lo:hi] = r // L; TL[lo:hi] = r % L; R[lo:hi] = G[lo:hi] % self.rows; RT[lo:hi] = (G[lo:hi] // self.rows) * L + r % L
        V[0][lo:hi] = u % self.nout            # the act value, (lane TL) -- read through TL below
        lo, hi = self.CELL0, self.PSG0; u = ids[lo:hi] - lo; kind[lo:hi] = 3; cs, w = u // self.nw, u % self.nw
        S[lo:hi] = cs // self.ncols; C[lo:hi] = cs % self.ncols; Wv[lo:hi] = w
        lo, hi = self.PSG0, self.PM0; u = ids[lo:hi] - lo; kind[lo:hi] = 4; g, gd = u // (self.ngroups + 1), u % (self.ngroups + 1)
        G[lo:hi] = g; SW[lo:hi] = np.where(gd == self.ngroups, 0, self.VEC0 + gd * self.NV); VALID[lo:hi] = (gd < self.ngroups)
        lo, hi = self.PM0, self.PSH0; u = ids[lo:hi] - lo; kind[lo:hi] = 5; cl = u % (self.ncmax + 1); lfg = u // (self.ncmax + 1)
        G[lo:hi] = lfg % self.ngroups; lf = lfg // self.ngroups; PL[lo:hi] = lf // self.fan; PF[lo:hi] = lf % self.fan; PCL[lo:hi] = cl
        if self.smax:
            lo, hi = self.PSH0, n; u = ids[lo:hi] - lo; kind[lo:hi] = 6; s = u % (self.smax + 1); lg = u // (self.smax + 1)
            G[lo:hi] = lg % self.ngroups; PL[lo:hi] = lg // self.ngroups; PS[lo:hi] = s
        return NS(kind=kind, G=G, R=R, TL=TL, RT=RT, S=S, C=C, W=Wv, SW=SW, VALID=VALID, V=V, PL=PL, PF=PF, PCL=PCL, PS=PS)


def build_vec(sp: VSpec):
    g = Circ(); one = g.one
    KS, NC, AB, WB, L, NG = sp.rows, sp.ncols, sp.ab, sp.wb, sp.lanes, sp.ngroups
    assert KS & (KS - 1) == 0 and KS <= 128 and NG <= 128, "rows power of two <= 128 and at most 128 register groups"
    QROWS = KS if (KS.bit_length() - 1) % 2 else 2 * KS
    assert QROWS <= 128
    F = KS * sp.fan
    assert F <= 256, "sum of fan*rows counts must stay bf16-exact (<= 256) for the shift mux"
    ref = {0: A_REF * one, 1: A_REF * one}
    T = sp.table(); kd = T.kind
    feat = lambda name, arr: g.token_feature(name, lambda t, a=arr: a[t])
    isv, isa = kd == 1, kd == 2

    # ---------------------------------------------------------------- token features (all static functions of the token id)
    row_arr = np.where(isv | isa, T.R, -1)
    arow = [feat(f"arow{s}", (row_arr == s).astype(np.int8)) for s in range(KS)]
    wr = [feat(f"wr{t}", np.where(isv, (T.G // KS) == (t // L), isa & (T.RT == t)).astype(np.int8)) for t in range(sp.tiles)]
    vbit = [[feat(f"vbit{l}.{a}", np.where(isv, (T.V[l] >> a) & 1, np.where(isa & (T.TL == l), (T.V[0] >> a) & 1, 0)).astype(np.int8))
             for a in range(AB)] for l in range(L)]
    gq = [feat(f"gq{x}", (isv & (T.G == x)).astype(np.int8)) for x in range(NG)]
    crow = [feat(f"crow{s}", ((kd == 3) & (T.S == s)).astype(np.int8)) for s in range(KS)]
    ccol = [feat(f"ccol{c}", ((kd == 3) & (T.C == c)).astype(np.int8)) for c in range(NC)]
    wbit = [feat(f"wbit{b}", np.where(kd == 3, (T.W >> b) & 1, 0).astype(np.int8)) for b in range(WB)]

    def mem_head(name, qfeat, kfeat, vfeats):
        """Write at PROG tokens (key one-hot kfeat), read at VEC tokens (query one-hot qfeat); value bits = vfeats."""
        nb = len(vfeats); assert nb <= 126
        if nb == 0: return []
        val = {**ref, **{2 + b: A_BIT * v for b, v in enumerate(vfeats)}}
        outs = [(2 + b, 2.0 ** 7) for b in range(nb)]
        return g.head(name, {i: K * q for i, q in enumerate(qfeat)}, {i: K * k for i, k in enumerate(kfeat)},
                      val, 0, {col: K * one for col, _ in outs}, outs)

    # ---------------------------------------------------------------- program memory (level 1)
    CB = sp.code_bits
    sk = [feat(f"sk{x}", ((kd == 4) & (T.G == x)).astype(np.int8)) for x in range(NG)]
    sv = [feat(f"sv{b}", np.where(kd == 4, (T.SW >> b) & 1, 0).astype(np.int8)) for b in range(CB)] + \
         [feat("svalid", np.where(kd == 4, T.VALID, 0).astype(np.int8))]
    out = mem_head("SUCC", gq, sk, sv)
    succ_bits, valid = out[:CB], out[CB]
    mval = [feat(f"mval{c}", ((kd == 5) & (T.PCL == c)).astype(np.int8)) for c in range(sp.ncmax)]
    mask_dims = []                           # mask_dims[l][f][cl]
    for l in range(L):
        row = []
        for fi in range(sp.fan):
            kf = [feat(f"mk{l}.{fi}.{x}", ((kd == 5) & (T.PL == l) & (T.PF == fi) & (T.G == x)).astype(np.int8)) for x in range(NG)]
            row.append(mem_head(f"MASK{l}.{fi}", gq, kf, mval[:len(sp.cls_cols[l])]))
        mask_dims.append(row)
    shd = [None] * L
    if sp.smax:
        shv = [feat(f"shv{s}", ((kd == 6) & (T.PS == s)).astype(np.int8)) for s in range(sp.smax + 1)]
        for l in range(L):
            kf = [feat(f"shk{l}.{x}", ((kd == 6) & (T.PL == l) & (T.G == x)).astype(np.int8)) for x in range(NG)]
            shd[l] = mem_head(f"SHIFT{l}", gq, kf, shv)

    # ---------------------------------------------------------------- weights, products, accumulators
    wbits = []
    for c in range(NC):
        key = {s: K * (crow[s] + ccol[c] - one) for s in range(KS)}
        qry = {s: K * arow[s] for s in range(KS)}
        val = {**ref, **{2 + b: A_BIT * wbit[b] for b in range(WB)}}
        outs = [(2 + b, 2.0 ** 7) for b in range(WB)]
        wbits.append(g.head(f"W{c}", qry, key, val, 0, {col: K * one for col, _ in outs}, outs))
    chans = [(a, b) for a in range(AB) for b in range(WB)]
    scale = 2.0 ** (7 + KS.bit_length() - 1)
    cnt = []
    for c in range(NC):
        t_c = sp.col_tile(c); lt = t_c % L; lam = sp.ccls[c]; cl = sp.cl_index[c]
        prod = [[g.ffn([(1, one, vbit[lt][a] + wbits[c][b] - one)], f"p{c}.{a}{b}") for b in range(WB)] for a in range(AB)]
        mask_c = sum((mask_dims[lam][fi][cl] for fi in range(sp.fan)), Expr())
        key = {s: K * arow[s] for s in range(KS)}
        qry = {s: K * mask_c for s in range(QROWS)}
        val = {**ref, **{2 + n: A_BIT * prod[a][b] for n, (a, b) in enumerate(chans)}}
        outs = [(2 + n, scale) for n in range(len(chans))]
        cnt.append(g.head(f"A{c}", qry, key, val, 0, {col: K * one for col, _ in outs}, outs, beta=wr[t_c]))

    # ---------------------------------------------------------------- adder tree + requantizer, once per lane
    sgn = lambda b: -1 if b == WB - 1 else 1

    def rng(p):
        pairs = [(a, p - a) for a in range(AB) if 0 <= p - a < WB]
        return -F * sum(1 for a, b in pairs if b == WB - 1), F * sum(1 for a, b in pairs if b != WB - 1)

    P0 = AB + WB - 1; sm = sp.smax
    rqs = []
    for l in range(L):
        Sab = {(a, b): sum((cnt[c][a * WB + b] for c in sp.cls_cols[l]), Expr()) for a in range(AB) for b in range(WB)}
        digits = []
        if not sm:
            for p in range(P0):
                e = sum((Sab[(a, p - a)] * sgn(p - a) for a in range(AB) if 0 <= p - a < WB), Expr())
                lo, hi = rng(p); digits.append((e, lo, hi))
            rqs.append(N.requant(g, digits, shift=0, amax=sp.amax, name=f"rq{l}"))
        else:
            for pp in range(P0 + sm):
                units = []
                for s in range(sm + 1):
                    p = pp - (sm - s)
                    if not 0 <= p < P0: continue
                    for a in range(AB):
                        b = p - a
                        if 0 <= b < WB: units.append((sgn(b), Sab[(a, b)], shd[l][s]))
                lo = min([0] + [rng(pp - (sm - s))[0] for s in range(sm + 1) if 0 <= pp - (sm - s) < P0])
                hi = max([0] + [rng(pp - (sm - s))[1] for s in range(sm + 1) if 0 <= pp - (sm - s) < P0])
                digits.append((g.node(units, f"mux{l}.{pp}"), lo, hi))
            rqs.append(N.requant(g, digits, shift=sm, amax=sp.amax, name=f"rq{l}"))

    # ---------------------------------------------------------------- next-token code
    ids = np.arange(sp.nvocab)
    code = [feat(f"code{i}", (128 * (2 * ((ids >> i) & 1) - 1)).astype(np.int16)) for i in range(CB)]
    is_run = feat("is_run", isv.astype(np.int8))
    for i in range(CB):
        nxt = [(1, one, succ_bits[i])]
        if (sp.NOP >> i) & 1: nxt.append((1, one, one - is_run))
        if i < L * AB: nxt.append((1, rqs[i // AB].bits[i % AB], valid))
        g.ffn([(2 * S_CODE * c, a, b) for c, a, b in nxt] + [(-S_CODE, one, one), (-1, one, code[i]), (1, one, -1 * code[i])],
              f"code{i}", into=code[i])
    return g, {"code": code, "rq": rqs, "cnt": cnt, "valid": valid}


# ====================================================================================================================== program
@dataclass
class OpGroup:
    dest: int          # destination group
    lanes: list        # per lane: (column ids, shift) or None


@dataclass
class VProgram:
    sp: VSpec
    W: dict = field(default_factory=dict)       # (row, column) -> weight
    groups: list = field(default_factory=list)  # OpGroups in execution order
    init: dict = field(default_factory=dict)    # flat register -> initial value (others 0)
    entry: int = 0                              # group whose VEC token is the last prompt token

    def check(self):
        sp = self.sp
        dests = [og.dest for og in self.groups]
        assert len(set(dests)) == len(dests) and self.entry not in dests, "SSA: one op group per destination group"
        for og in self.groups:
            assert 0 <= og.dest < sp.ngroups and len(og.lanes) == sp.lanes
            for l, op in enumerate(og.lanes):
                if op is None: continue
                cols, sh = op
                assert len(cols) <= sp.fan and 0 <= sh <= sp.smax and len(set(cols)) == len(cols)
                assert all(sp.ccls[c] == l for c in cols), "a lane's op may only read columns of its own class"
        for (s, c), w in self.W.items(): assert 0 <= s < sp.rows and 0 <= c < sp.ncols and sp.wlo <= w <= sp.whi
        return self

    def prompt(self):
        sp = self.sp; self.check(); p = []
        p += [sp.cell(s, c, w) for (s, c), w in sorted(self.W.items()) if w != 0]
        chain = [self.entry] + [og.dest for og in self.groups]
        for k, og in enumerate(self.groups):
            pred = chain[k]; p.append(sp.psg(pred, og.dest))
            for l, op in enumerate(og.lanes):
                cols, sh = ([], 0) if op is None else op
                for f in range(sp.fan):
                    p.append(sp.pm(l, f, pred, sp.cl_index[cols[f]] if f < len(cols) else sp.ncmax))
                if sp.smax: p.append(sp.psh(l, pred, sh))
        p.append(sp.psg(chain[-1], sp.ngroups))
        for r in range(sp.nreg):
            if sp.reg_group(r) != self.entry: p.append(sp.act(r, self.init.get(r, 0)))
        p.append(sp.vec(self.entry, [self.init.get(self.entry * sp.lanes + l, 0) for l in range(sp.lanes)]))
        return p

    def expected(self):
        return reference_emit(self.sp, self.prompt())


def reference_emit(sp: VSpec, prompt, max_new=100000):
    """Independent numpy model of the vectorized machine: parses the prompt, applies each token's register writes, then
    computes the successor group.  Returns the emitted tokens (excluding the final HALT)."""
    L = sp.lanes
    W = np.zeros((sp.rows, sp.ncols), dtype=np.int64); a = np.zeros(sp.nreg, dtype=np.int64)
    succ, mask, shift = {}, {}, {}
    for t in prompt:
        if sp.is_cell(t): s, c, w = sp.cell_dec(t); W[s, c] = w
        elif sp.is_act(t): r, v = sp.act_dec(t); a[r] = v
        elif sp.is_vec(t):
            g, vs = sp.vec_dec(t)
            for l in range(L): a[g * L + l] = vs[l]
        else:
            d = sp.prog_dec(t)
            if d is None: continue
            if d[0] == "PSG": succ[d[1]] = d[2]
            elif d[0] == "PM": mask[(d[1], d[2], d[3])] = d[4]
            elif d[0] == "PSH": shift[(d[1], d[2])] = d[3]
    cur, out = prompt[-1], []
    while len(out) < max_new:
        if not sp.is_vec(cur): break
        g, _ = sp.vec_dec(cur)
        gd = succ.get(g, sp.ngroups)
        if gd == sp.ngroups: break
        vs = []
        for l in range(L):
            y = 0
            for f in range(sp.fan):
                cl = mask.get((l, f, g), sp.ncmax)
                if cl < len(sp.cls_cols[l]):
                    c = sp.cls_cols[l][cl]; t = sp.col_tile(c)
                    y += int(sum(int(a[sp.tile_reg(t, s)]) * int(W[s, c]) for s in range(sp.rows)))
            vs.append(int(np.clip(y >> shift.get((l, g), 0), 0, sp.amax)) if ((l, g) in shift or not sp.smax) else 0)
        nxt = sp.vec(gd, vs)
        out.append(nxt); cur = nxt
        for l in range(L): a[gd * L + l] = vs[l]
    return out


# ====================================================================================================================== scheduler
@dataclass
class VCompiled:
    spec: VSpec
    graph: object
    W: dict
    groups: list
    in_regs: list
    const_regs: list
    out_regs: list
    entry: int
    neurons: list = field(default_factory=list)

    def program(self, xq):
        xq = np.asarray(xq, dtype=np.int64).reshape(-1)
        assert len(xq) == len(self.in_regs) and (xq.size == 0 or (0 <= xq.min() and xq.max() <= self.spec.amax))
        init = {r: 1 for r in self.const_regs}
        init.update({r: int(v) for r, v in zip(self.in_regs, xq) if v})
        return VProgram(self.spec, dict(self.W), list(self.groups), init, self.entry)

    def decode(self, emitted):
        sp = self.spec; val = {}
        for t in emitted:
            if sp.is_vec(t):
                g, vs = sp.vec_dec(t)
                for l, v in enumerate(vs): val[g * sp.lanes + l] = v
        return np.array([val.get(r, 0) if r >= 0 else 0 for r in self.out_regs], dtype=np.int64)


def compile_vec(g, rows=16, lanes=2, spec_override=None, fan=None, smax=None):
    """qwentensor.ir.Graph -> VSpec + VProgram skeleton.  Neurons of one matvec node are paired into op groups (lane l = neuron
    index mod lanes within the node); each node starts on a fresh group.  `spec_override`: fit the graph onto an existing machine."""
    g.validate(machine=True)
    L = lanes; br = g.bias_rows; dr = rows - br; assert dr >= 1
    lo, hi = g.wlo, g.whi
    cur = [0]                                    # slot counter: slot s -> group index s // L (data groups only), lane s % L

    def align(): cur[0] = -(-cur[0] // L) * L

    def new_reg():
        gi, l = divmod(cur[0], L); cur[0] += 1
        return ((gi // dr) * rows + gi % dr) * L + l

    vals, neurons, in_regs = {}, [], []
    for node in g.nodes:
        if node.op == "input":
            align(); vals[node.name] = [new_reg() for _ in range(node.out_len)]; in_regs = vals[node.name]
        elif node.op == "gather":
            src = vals[node.inputs[0]]; vals[node.name] = [src[i] if i >= 0 else -1 for i in node.index]
        elif node.op == "concat":
            vals[node.name] = [r for i in node.inputs for r in vals[i]]
        else:
            x = vals[node.inputs[0]]; Kk, Nn = node.W.shape; out = []; align()
            for p in range(node.groups):
                xs = x[p * Kk:(p + 1) * Kk]
                for n in range(Nn):
                    wts = {}
                    for k in range(Kk):
                        w = int(node.W[k, n])
                        if w and xs[k] >= 0: wts[xs[k]] = wts.get(xs[k], 0) + w
                    wts = {r: w for r, w in wts.items() if w}
                    d = new_reg(); out.append(d)
                    neurons.append(dict(name=f"{node.name}[{p},{n}]", wts=wts, bias=int(node.b[n]), shift=node.shift, dest=d, cols=[]))
            vals[node.name] = out
    out_regs = vals[g.output]
    ngd = -(-cur[0] // L); nblk = max(1, -(-ngd // dr))
    if spec_override is not None:
        sp0 = spec_override
        if sp0.rows != rows or sp0.lanes != L or sp0.nblk < nblk or (sp0.ab, sp0.wb) != (g.ab, g.wb):
            raise DoesNotFit(f"needs rows={rows} lanes={L} blocks>={nblk}; machine has {sp0}")
        nblk = sp0.nblk
    ntile = nblk * L
    const_regs = [(b * rows + dr + i) * L + l for b in range(nblk) for i in range(br) for l in range(L)]
    rtile = lambda r: ((r // L) // rows) * L + r % L
    rrow = lambda r: (r // L) % rows
    # ---- columns
    pools = None
    if spec_override is not None:
        pools = {}
        for c in range(spec_override.ncols):
            pools.setdefault((spec_override.col_tile(c), spec_override.ccls[c]), []).append(c)
    used = [0] * ntile; cls_of = {}; W = {}
    for nu in neurons:
        lane = nu["dest"] % L; by_tile = {}
        for r, w in nu["wts"].items(): by_tile.setdefault(rtile(r), {})[rrow(r)] = _split(w, lo, hi)
        bparts = _split(nu["bias"], lo, hi)
        if len(bparts) > br: raise DoesNotFit(f"{nu['name']}: bias {nu['bias']} needs {len(bparts)} constant rows, bias_rows={br}")
        btile = min(by_tile) if by_tile else (nu["dest"] // L // rows) * L          # bias rows are constant-1 in every tile
        if bparts: by_tile.setdefault(btile, {})
        for t, rw in sorted(by_tile.items()):
            k = max([len(v) for v in rw.values()] + [1]); got = []
            for _ in range(k):
                if pools is None:
                    got.append(used[t]); cls_of[(t, used[t])] = lane; used[t] += 1
                else:
                    pl = pools.get((t, lane), [])
                    if not pl: raise DoesNotFit(f"{nu['name']}: no free column of lane class {lane} in tile {t}")
                    got.append(pl.pop(0) - spec_override.col_base(t))
            for j, ci in enumerate(got):
                nu["cols"].append((t, ci))
                for s_, parts in rw.items():
                    if j < len(parts): W[(s_, (t, ci))] = parts[j]
                if bparts and t == btile and j == 0:
                    for i, bp in enumerate(bparts): W[(dr + i, (t, ci))] = bp
    fan_need = max([len(nu["cols"]) for nu in neurons] + [1]); smax_need = max([nu["shift"] for nu in neurons] + [0])
    if spec_override is None:
        sp = VSpec(rows=rows, lanes=L, cols=tuple(used), ccls=tuple(cls_of[(t, i)] for t in range(ntile) for i in range(used[t])),
                   ab=g.ab, wb=g.wb, fan=max(fan or 0, fan_need), smax=max(smax or 0, smax_need))
    else:
        sp = spec_override
        if fan_need > sp.fan or smax_need > sp.smax: raise DoesNotFit(f"needs fan={fan_need} smax={smax_need}; machine has {sp}")
    if sp.ngroups > 128: raise DoesNotFit(f"{sp.ngroups} register groups > 128 (one mask/shift head holds 128 slots)")
    cid = lambda tc: sp.col_id(*tc)
    Wp = {(s, cid(tc)): w for (s, tc), w in W.items()}
    groups = []; by_dest = {}
    for nu in neurons:
        gd = nu["dest"] // L; lane = nu["dest"] % L
        if gd not in by_dest:
            by_dest[gd] = OpGroup(gd, [None] * L); groups.append(by_dest[gd])
        by_dest[gd].lanes[lane] = ([cid(tc) for tc in nu["cols"]], nu["shift"])
    return VCompiled(sp, g, Wp, groups, in_regs, const_regs, out_regs, const_regs[0] // L, neurons)


class VecMachine:
    """The vectorized circuit evaluated by the exact float64 IR evaluator (teacher forcing on the model's own tokens)."""

    def __init__(self, sp):
        self.sp = sp; self.g, self.d = build_vec(sp)

    def next_tokens(self, toks, start=0):
        m = ir.evaluate(self.g, toks)
        cd = self.d["code"]
        return [sum((1 << i) * int(ir.ev(cd[i], m[t:t + 1])[0] > 0) for i in range(len(cd))) for t in range(start, len(toks))]

    def run(self, prompt, emitted): return self.next_tokens(list(prompt) + list(emitted), len(prompt) - 1)
    def stats(self): return {**self.g.stats(), "layers": self.g.n_layers()}


def install_vec(sp):
    """Make qwentpu.merge.build_merged / qwentpu.driver.TpuSession build and feed THIS machine."""
    from qwentpu import circuit as C
    C.build = lambda: build_vec(sp)
    C.VOCAB = list(range(sp.nvocab)); C.CODE = sp.code_bits
    return sp
