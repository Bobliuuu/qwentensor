"""The executor circuit in qwentpu's IR.

Datapath (same cells as qwentpu/circuit.py, regrouped):
  * weight heads W[c]   : key = (row s) via CELL(s,c,w); read at ACT(r;v) with query = row s -> the bits of w[s][c]
  * AND products        : p[c][a][b] = act bit a AND weight bit b
  * accumulator heads A[c]: key = row s, written by ACT(r;v) ONLY IF r belongs to column c's tile (write gate beta),
                          queried (all rows at once) iff the program-memory mask of the READ op names column c
  * program memory      : succ / mask / shift heads keyed by the op slot or register of the CURRENT token, written
                          by PROG tokens in the prompt, read at level 1
  * adder tree          : counts of all columns in the mask are summed (K-tiling), optional one-hot shift mux on the
                          digits, then nonlin.requant (exact floor shift + relu + clamp)
  * code                : next token = succ[slot(current)]  +  requantised value bits (READ ops only)
"""
from qwentpu.compiler import nonlin as N
from qwentpu.compiler.gadgets import Circ
from qwentpu.compiler.ir import A_BIT, A_REF, K, Expr

from .spec import Spec
from .wide import normalize

S_CODE = 2 ** 12


def build(sp: Spec):
    g = Circ()
    one, f = g.one, g.token_feature
    KS, NC, AB, WB, NOUT, NOPS = sp.rows, sp.ncols, sp.ab, sp.wb, sp.nout, sp.nops
    assert KS & (KS - 1) == 0 and KS <= 128 and NOPS <= 128
    QROWS = KS if (KS.bit_length() - 1) % 2 else 2 * KS      # multi-hot query size n with n*128 a power of 4
    assert QROWS <= 128, "rows=64 needs 128 query rows ... unsupported"
    F = KS * sp.gf
    assert F <= 256, "sum of fan*rows counts must stay bf16-exact (<= 256) for the shift mux"
    ref = {0: A_REF * one, 1: A_REF * one}

    # ---------------------------------------------------------------- token features
    T = sp.table(); ACT, READ, CELL, PSR, PSA, PM, PSH = 1, 2, 3, 4, 5, 6, 7
    kd, TA, TB, TC = T.kind, T.A, T.B, T.C
    is_act = f("is_act", lambda t: int(kd[t] == ACT))
    is_read = f("is_read", lambda t: int(kd[t] == READ))
    crow = [f(f"crow{s}", lambda t, s=s: int(kd[t] == CELL and TA[t] == s)) for s in range(KS)]
    ccol = [f(f"ccol{c}", lambda t, c=c: int(kd[t] == CELL and TB[t] == c)) for c in range(NC)]
    wbit = [f(f"wbit{b}", lambda t, b=b: (int(TC[t]) >> b) & 1 if kd[t] == CELL else 0) for b in range(WB)]
    arow = [f(f"arow{s}", lambda t, s=s: int(kd[t] == ACT and TA[t] % KS == s)) for s in range(KS)]
    atile = [f(f"atile{k}", lambda t, k=k: int(kd[t] == ACT and TA[t] // KS == k)) for k in range(sp.tiles)]
    abit = [f(f"abit{a}", lambda t, a=a: (int(TB[t]) >> a) & 1 if kd[t] == ACT else 0) for a in range(AB)]

    # ---------------------------------------------------------------- program memory (level 1)
    def mem_head(name, qfeat, kfeat, vfeats):
        """Write at PROG tokens (key one-hot kfeat), read at run tokens (query one-hot qfeat); value bits = vfeats."""
        nb = len(vfeats)
        assert nb <= 126
        val = {**ref, **{2 + b: A_BIT * v for b, v in enumerate(vfeats)}}
        outs = [(2 + b, 2.0 ** 7) for b in range(nb)]
        return g.head(name, {i: K * q for i, q in enumerate(qfeat)}, {i: K * k for i, k in enumerate(kfeat)},
                      val, 0, {col: K * one for col, _ in outs}, outs)

    CB = sp.code_bits
    succ_bits = [Expr() for _ in range(CB)]
    for h in range(sp.nsucc_heads):
        qf = [f(f"sq{h}.{i}", lambda t, i=i, h=h: int(T.slot_read[t] == 128 * h + i)) for i in range(128)]
        kf = [f(f"sk{h}.{i}", lambda t, i=i, h=h: int(T.slot_write[t] == 128 * h + i)) for i in range(128)]
        vf = [f(f"sv{h}.{b}", lambda t, b=b: (int(T.succ_word[t]) >> b) & 1) for b in range(CB)]
        out = mem_head(f"SUCC{h}", qf, kf, vf)
        succ_bits = [s + o for s, o in zip(succ_bits, out)]

    opq = [f(f"opq{j}", lambda t, j=j: int(kd[t] == READ and TA[t] == j)) for j in range(NOPS)]
    mval = [f(f"mval{c}", lambda t, c=c: int(kd[t] == PM and TC[t] == c)) for c in range(NC)]
    mask_dims = []                      # mask_dims[f][c]
    for fi in range(sp.fan):
        kf = [f(f"mk{fi}.{j}", lambda t, j=j, fi=fi: int(kd[t] == PM and TA[t] == fi and TB[t] == j)) for j in range(NOPS)]
        mask_dims.append(mem_head(f"MASK{fi}", opq, kf, mval))
    shd = None
    if sp.smax:
        kf = [f(f"shk{j}", lambda t, j=j: int(kd[t] == PSH and TA[t] == j)) for j in range(NOPS)]
        sv = [f(f"shv{s}", lambda t, s=s: int(kd[t] == PSH and TB[t] == s)) for s in range(sp.smax + 1)]
        shd = mem_head("SHIFT", opq, kf, sv)

    # ---------------------------------------------------------------- weights, products, accumulators
    wbits = []
    for c in range(NC):
        key = {s: K * (crow[s] + ccol[c] - one) for s in range(KS)}
        qry = {s: K * arow[s] for s in range(KS)}
        val = {**ref, **{2 + b: A_BIT * wbit[b] for b in range(WB)}}
        outs = [(2 + b, 2.0 ** 7) for b in range(WB)]
        wbits.append(g.head(f"W{c}", qry, key, val, 0, {col: K * one for col, _ in outs}, outs))
    prod = [[[g.ffn([(1, one, abit[a] + wbits[c][b] - one)], f"p{c}.{a}{b}") for b in range(WB)] for a in range(AB)]
            for c in range(NC)]
    chans = [(a, b) for a in range(AB) for b in range(WB)]
    scale = 2.0 ** (7 + KS.bit_length() - 1)
    cnt = []
    for c in range(NC):
        t_c = sp.col_tile(c)
        mask_c = sum((mask_dims[fi][c] for fi in range(sp.fan)), Expr())
        key = {s: K * arow[s] for s in range(KS)}
        qry = {s: K * mask_c for s in range(QROWS)}
        val = {**ref, **{2 + n: A_BIT * prod[c][a][b] for n, (a, b) in enumerate(chans)}}
        outs = [(2 + n, scale) for n in range(len(chans))]
        cnt.append(g.head(f"A{c}", qry, key, val, 0, {col: K * one for col, _ in outs}, outs, beta=atile[t_c]))

    # ---------------------------------------------------------------- adder tree
    neg = 1 << (WB - 1)
    G = sp.ngroups
    Sg = [{(a, b): sum((cnt[c][a * WB + b] for c in range(NC) if sp.col_group(c) == gm), Expr()) for a in range(AB) for b in range(WB)}
          for gm in range(G)]
    Sab = Sg[0]
    sgn = lambda b: -1 if b == WB - 1 else 1

    def rng(p):
        """range of digit position p = a+b (weight 2^p) before any shift: sum of +-S_ab, S_ab in [0, F]."""
        pairs = [(a, p - a) for a in range(AB) if 0 <= p - a < WB]
        return -F * sum(1 for a, b in pairs if b == WB - 1), F * sum(1 for a, b in pairs if b != WB - 1)

    P0 = AB + WB - 1
    sm = sp.smax
    digits = []
    if G > 1:
        # several groups (docs/WIDE.md): each group's accumulator is normalised on its own, the tiny digits are summed over the groups
        comb = {}
        for gm in range(G):
            dg = []
            for p in range(P0):
                e = sum((Sg[gm][(a, p - a)] * sgn(p - a) for a in range(AB) if 0 <= p - a < WB), Expr())
                lo, hi = rng(p); dg.append((e, lo, hi))
            for pos, (e, lo, hi) in normalize(g, dg, f"norm{gm}").items():
                x = comb.get(pos); comb[pos] = (e, lo, hi) if x is None else (x[0] + e, x[1] + lo, x[2] + hi)
        top = max(comb); D = [comb.get(j, (0 * one, 0, 0)) for j in range(top + 1)]
        if not sm:
            rq = N.requant(g, D, shift=0, amax=sp.amax, name="rq")
        else:
            for pp in range(top + 1 + sm):
                units = []; rs = []
                for s in range(sm + 1):
                    p = pp - (sm - s)
                    if 0 <= p <= top and D[p][0] is not None and (D[p][1], D[p][2]) != (0, 0):
                        units.append((1, D[p][0], shd[s])); rs.append(D[p][1:])
                lo = min([0] + [r[0] for r in rs]); hi = max([0] + [r[1] for r in rs])
                digits.append((g.node(units, f"mux{pp}"), lo, hi))
            rq = N.requant(g, digits, shift=sm, amax=sp.amax, name="rq")
    elif not sm:
        for p in range(P0):
            e = sum((Sab[(a, p - a)] * sgn(p - a) for a in range(AB) if 0 <= p - a < WB), Expr())
            lo, hi = rng(p); digits.append((e, lo, hi))
        rq = N.requant(g, digits, shift=0, amax=sp.amax, name="rq")
    else:
        # y' = y * 2^(smax - s): digit position p' = a+b + smax - s, selected by the one-hot shift dims; then a FIXED
        # floor-shift by smax gives floor(y / 2^s).  Selecting counts (<= F <= 256, bf16 exact) costs one FFN level.
        for pp in range(P0 + sm):
            units = []
            for s in range(sm + 1):
                p = pp - (sm - s)
                if not 0 <= p < P0: continue
                for a in range(AB):
                    b = p - a
                    if 0 <= b < WB: units.append((sgn(b), Sab[(a, b)], shd[s]))
            # exactly one shift is selected (or none): the envelope is the union, not the sum
            lo = min([0] + [rng(pp - (sm - s))[0] for s in range(sm + 1) if 0 <= pp - (sm - s) < P0])
            hi = max([0] + [rng(pp - (sm - s))[1] for s in range(sm + 1) if 0 <= pp - (sm - s) < P0])
            digits.append((g.node(units, f"mux{pp}"), lo, hi))
        rq = N.requant(g, digits, shift=sm, amax=sp.amax, name="rq")

    # ---------------------------------------------------------------- next-token code
    code = [f(f"code{i}", lambda t, i=i: 128 * (2 * ((t >> i) & 1) - 1)) for i in range(CB)]
    is_run = is_act + is_read
    for i in range(CB):
        nxt = [(1, one, succ_bits[i])]
        if (sp.NOP >> i) & 1: nxt.append((1, one, one - is_run))
        if i < AB: nxt.append((1, rq.bits[i], is_read))
        g.ffn([(2 * S_CODE * c, a, b) for c, a, b in nxt] + [(-S_CODE, one, one), (-1, one, code[i]), (1, one, -1 * code[i])],
              f"code{i}", into=code[i])
    return g, {"code": code, "rq": rq, "cnt": cnt, "wbits": wbits, "mask": mask_dims, "succ": succ_bits}
