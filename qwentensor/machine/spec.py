"""Machine configuration and token layout.

Register file: `tiles` banks of `rows` registers.  Register id r = tile * rows + s.  Row `rows-1` of every tile is the
constant-1 register (bias); `bias_rows` of the top rows may be reserved by the scheduler.
Columns ("neurons"): `cols[t]` physical columns in tile t.  A column holds one weight per row of ITS tile (a weight head
keyed by row, an accumulator head keyed by row).  An op (READ slot j) sums up to `fan` columns (any tiles), so one logical
neuron may read up to fan * (rows-1) registers: that is the K-tiling, inside one exact adder tree.

Token ids (small ids first so the emitted tokens need few code bits):
  0 HALT   1 NOP   ACT0 .. : ACT(r; v) = ACT0 + r*NOUT + v   (ACT0 is a multiple of NOUT: no carry when adding v)
  READ(j) = READ0 + j                       (j < nops: the op slot, i.e. the "program counter" values)
  CELL(s,c,w)                               weight write (prompt only)
  PS_READ(j,r)  succ[slot j]  = ACT(r;0)    program-memory write: op j's result goes to register r
  PS_ACT(r,j)   succ[slot r]  = READ(j) | HALT (j == nops)     register r's successor is op j
  PM(f,j,c)     mask field f of op j := column c (c == ncols: none)
  PSH(j,s)      shift of op j := s
"""
from dataclasses import dataclass
from functools import cached_property


GROUP_F = 48     # fan*rows one adder-tree group may sum (the requantizer's carry staircase must fit its 256 gate window, docs/WIDE.md)


def group_fan(rows): return max(1, GROUP_F // rows)


@dataclass(frozen=True)
class Spec:
    rows: int = 8                  # registers per tile (power of two, <= 128); last row = constant 1
    cols: tuple = (8,)             # physical columns per tile
    ab: int = 3
    wb: int = 3
    nops: int = 16                 # op slots (program length bound)
    fan: int = 1                   # max columns summed by one op
    smax: int = 0                  # max requant shift (0: shift fixed to 0, no mux)
    tpg: int = 0                   # tiles per adder-tree group (0: one group = the legacy single tree); docs/WIDE.md
    gfan: int = 0                  # max active columns of one op inside one group (0: = fan)

    # ---- derived ----
    @cached_property
    def tiles(self): return len(self.cols)
    @cached_property
    def ncols(self): return sum(self.cols)
    @cached_property
    def ngroups(self): return 1 if not self.tpg else -(-self.tiles // self.tpg)
    @cached_property
    def gf(self): return self.gfan or self.fan
    def col_group(self, c): return 0 if not self.tpg else self.col_tile(c) // self.tpg
    @cached_property
    def nreg(self): return self.tiles * self.rows
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
    def nslots(self): return self.nops + self.nreg
    @cached_property
    def nsucc_heads(self): return -(-self.nslots // 128)

    def col_tile(self, c):
        for t, n in enumerate(self.cols):
            if c < n: return t
            c -= n
        raise IndexError
    def col_base(self, t): return sum(self.cols[:t])
    def col_id(self, t, i):
        assert i < self.cols[t]; return self.col_base(t) + i

    # ---- token layout ----
    HALT = 0
    NOP = 1
    @cached_property
    def ACT0(self): return self.nout
    @cached_property
    def READ0(self): return self.ACT0 + self.nreg * self.nout
    @cached_property
    def CELL0(self): return self.READ0 + self.nops
    @cached_property
    def PSR0(self): return self.CELL0 + self.rows * self.ncols * self.nw
    @cached_property
    def PSA0(self): return self.PSR0 + self.nops * self.nreg
    @cached_property
    def PM0(self): return self.PSA0 + self.nreg * (self.nops + 1)
    @cached_property
    def PSH0(self): return self.PM0 + self.fan * self.nops * (self.ncols + 1)
    @cached_property
    def nvocab(self): return self.PSH0 + (self.nops * (self.smax + 1) if self.smax else 0)
    @cached_property
    def code_bits(self): return max(8, (self.nvocab - 1).bit_length())

    def act(self, r, v): return self.ACT0 + r * self.nout + int(v)
    def read(self, j): return self.READ0 + j
    def cell(self, s, c, w): return self.CELL0 + (s * self.ncols + c) * self.nw + (int(w) & (self.nw - 1))
    def ps_read(self, j, r): return self.PSR0 + j * self.nreg + r
    def ps_act(self, r, j): return self.PSA0 + r * (self.nops + 1) + j
    def pm(self, f, j, c): return self.PM0 + (f * self.nops + j) * (self.ncols + 1) + c
    def psh(self, j, s): return self.PSH0 + j * (self.smax + 1) + s

    # ---- classifiers / decoders ----
    def is_act(self, t): return self.ACT0 <= t < self.READ0
    def is_read(self, t): return self.READ0 <= t < self.CELL0
    def is_cell(self, t): return self.CELL0 <= t < self.PSR0
    def act_reg(self, t): return (t - self.ACT0) // self.nout
    def act_val(self, t): return (t - self.ACT0) % self.nout
    def cell_dec(self, t):
        u, w = divmod(t - self.CELL0, self.nw); s, c = divmod(u, self.ncols)
        return s, c, (w - self.nw if w >= self.nw // 2 else w)
    def slot(self, t):
        """Program-memory address of the CURRENT token (read side): op j for READ(j), nops + r for ACT(r;.), else -1."""
        if self.is_read(t): return t - self.READ0
        if self.is_act(t): return self.nops + self.act_reg(t)
        return -1
    def prog_dec(self, t):
        """Decode a PROG token -> (kind, a, b, c) or None."""
        if self.PSR0 <= t < self.PSA0:
            j, r = divmod(t - self.PSR0, self.nreg); return ("PSR", j, r)
        if self.PSA0 <= t < self.PM0:
            r, j = divmod(t - self.PSA0, self.nops + 1); return ("PSA", r, j)
        if self.PM0 <= t < self.PSH0:
            u, c = divmod(t - self.PM0, self.ncols + 1); f, j = divmod(u, self.nops); return ("PM", f, j, c)
        if self.smax and self.PSH0 <= t < self.nvocab:
            j, s = divmod(t - self.PSH0, self.smax + 1); return ("PSH", j, s)
        return None
    def succ_word(self, t):
        """Stored successor id of a PS_READ / PS_ACT token."""
        d = self.prog_dec(t)
        if d[0] == "PSR": return self.act(d[2], 0)
        j = d[2]; return self.HALT if j == self.nops else self.read(j)
    def succ_slot(self, t):
        d = self.prog_dec(t)
        return d[1] if d[0] == "PSR" else self.nops + d[1]

    def table(self):
        """Per-token-id numpy decode table (vectorised; the circuit's token features index it)."""
        import numpy as np
        n = self.nvocab
        kind = np.zeros(n, np.int8); A = np.full(n, -1, np.int32); B = np.full(n, -1, np.int32); C = np.full(n, -1, np.int32)
        sw = np.zeros(n, np.int64); srd = np.full(n, -1, np.int32); swr = np.full(n, -1, np.int32)
        ids = np.arange(n)
        def seg(lo, hi, k):
            kind[lo:hi] = k; return ids[lo:hi] - lo
        u = seg(self.ACT0, self.READ0, 1); A[self.ACT0:self.READ0] = u // self.nout; B[self.ACT0:self.READ0] = u % self.nout
        srd[self.ACT0:self.READ0] = self.nops + A[self.ACT0:self.READ0]
        u = seg(self.READ0, self.CELL0, 2); A[self.READ0:self.CELL0] = u; srd[self.READ0:self.CELL0] = u
        u = seg(self.CELL0, self.PSR0, 3); cs, w = u // self.nw, u % self.nw
        A[self.CELL0:self.PSR0] = cs // self.ncols; B[self.CELL0:self.PSR0] = cs % self.ncols; C[self.CELL0:self.PSR0] = w
        u = seg(self.PSR0, self.PSA0, 4); j, r = u // self.nreg, u % self.nreg
        A[self.PSR0:self.PSA0] = j; B[self.PSR0:self.PSA0] = r; swr[self.PSR0:self.PSA0] = j
        sw[self.PSR0:self.PSA0] = self.ACT0 + r * self.nout
        u = seg(self.PSA0, self.PM0, 5); r, j = u // (self.nops + 1), u % (self.nops + 1)
        A[self.PSA0:self.PM0] = r; B[self.PSA0:self.PM0] = j; swr[self.PSA0:self.PM0] = self.nops + r
        sw[self.PSA0:self.PM0] = np.where(j == self.nops, 0, self.READ0 + j)
        u = seg(self.PM0, self.PSH0, 6); c = u % (self.ncols + 1); fj = u // (self.ncols + 1)
        A[self.PM0:self.PSH0] = fj // self.nops; B[self.PM0:self.PSH0] = fj % self.nops; C[self.PM0:self.PSH0] = c
        if self.smax:
            u = seg(self.PSH0, n, 7); A[self.PSH0:n] = u // (self.smax + 1); B[self.PSH0:n] = u % (self.smax + 1)
        from types import SimpleNamespace as NS
        return NS(kind=kind, A=A, B=B, C=C, succ_word=sw, slot_read=srd, slot_write=swr)
