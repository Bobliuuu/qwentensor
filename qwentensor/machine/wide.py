"""Carry normalisation of one group's accumulator (docs/WIDE.md).

`normalize(g, digits)` rewrites y = sum 2^j d_j (digits with large ranges) as y = sum 2^j D_j with D_j in [0, 2^q - 1] on a few positions and a
top carry in [-1, 0].  Every step is a floor-division by 2^q whose staircase passes qwenTPU's 256-wide gate window test, so each is one exact FFN stage.
"""
from qwentpu.compiler.nonlin import _stair_ok

QMAX = 4


def qmax_for(G, cap=256):
    """Largest digit width q (<= QMAX) with G*(2^q - 1) <= cap: the combined digits are mux operands and must stay bf16-exact integers (<= 256)."""
    q = 1
    while q < QMAX and G * ((2 << q) - 1) <= cap: q += 1
    return q


def normalize(g, digits, name="norm", qmax=QMAX):
    """digits: [(Expr, lo, hi)] (y = sum 2^j digit_j).  Returns {position: (Expr, lo, hi)} with y = sum 2^pos Expr."""
    one = g.one; n = len(digits)
    dig = lambda j: digits[j] if j < n else (0 * one, 0, 0)
    c, clo, chi = 0 * one, 0, 0
    out = {}; j = 0; k = 0
    while j < n or (clo, chi) not in ((0, 0), (-1, 0), (-1, -1)):
        for q in range(qmax, 0, -1):
            parts = [dig(j + i) for i in range(q)]
            tlo = clo + sum(p[1] << i for i, p in enumerate(parts)); thi = chi + sum(p[2] << i for i, p in enumerate(parts))
            if _stair_ok(tlo, thi, 1 << q): break
        else:
            raise ValueError(f"group carry chain at bit {j} spans [{tlo},{thi}]: wider than the 256 gate window (fewer active columns per group)")
        t = c + sum((p[0] * (1 << i) for i, p in enumerate(parts)), 0 * one)
        D = 1 << q; base = tlo // D
        units = []
        for m in range(base + 1, thi // D + 1):
            units += g.terms_ge(t, m * D, tlo, thi)
        cn = g.node(units, f"{name}.c{k}") + base
        nlo, nhi = base, thi // D
        out[j] = (t - cn * D, 0, D - 1)
        c, clo, chi = cn, nlo, nhi; j += q; k += 1
    if (clo, chi) != (0, 0): out[j] = (c, clo, chi)
    return out
