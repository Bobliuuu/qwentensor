"""Low-level program for the register machine, its prompt, and an independent token-level numpy reference.

The PROMPT is the only host input: CELL tokens (weights), PROG tokens (the program memory image: per op its register
destination, column mask and shift; per register its successor op), ACT tokens (initial registers, the entry register
last).  Everything after it is emitted by the model.
"""
from dataclasses import dataclass, field

import numpy as np

from .spec import Spec


@dataclass
class Op:
    cols: list            # physical column ids summed by this op (len <= fan)
    shift: int            # requant shift
    dest: int             # destination register
    name: str = ""


@dataclass
class Program:
    sp: Spec
    W: dict = field(default_factory=dict)      # (row s, column c) -> weight (nonzero)
    ops: dict = field(default_factory=dict)    # slot -> Op
    order: list = field(default_factory=list)  # slots in execution order
    init: dict = field(default_factory=dict)   # register -> initial value (every other register starts at 0)
    entry: int = 0                             # register whose ACT is the last prompt token

    def check(self):
        sp = self.sp
        assert sorted(self.order) == sorted(self.ops), "order must list every op slot once"
        assert self.entry in self.init
        dests = [self.ops[j].dest for j in self.order]
        assert len(set(dests)) == len(dests), "SSA: one op per destination register (a register's successor is a property of the register)"
        assert not set(dests) & set(self.init)
        for j, o in self.ops.items():
            assert 0 <= j < sp.nops and len(o.cols) <= sp.fan and 0 <= o.shift <= max(sp.smax, 0)
            assert len(set(o.cols)) == len(o.cols) and all(0 <= c < sp.ncols for c in o.cols)
        for (s, c), w in self.W.items():
            assert 0 <= s < sp.rows and 0 <= c < sp.ncols and sp.wlo <= w <= sp.whi
        return self

    def prompt(self):
        sp = self.sp; self.check(); p = []
        p += [sp.cell(s, c, w) for (s, c), w in sorted(self.W.items()) if w != 0]
        for j, o in sorted(self.ops.items()):
            p.append(sp.ps_read(j, o.dest))
            for f in range(sp.fan):
                p.append(sp.pm(f, j, o.cols[f] if f < len(o.cols) else sp.ncols))
            if sp.smax: p.append(sp.psh(j, o.shift))
        chain = [self.entry] + [self.ops[j].dest for j in self.order]
        nxt = list(self.order) + [sp.nops]                    # nops == HALT
        for r, j in zip(chain, nxt): p.append(sp.ps_act(r, j))
        for r in range(sp.nreg):
            if r not in self.init: p.append(sp.act(r, 0))
        p += [sp.act(r, v) for r, v in sorted(self.init.items()) if r != self.entry]
        p.append(sp.act(self.entry, self.init[self.entry]))
        return p

    def expected(self):
        return reference_emit(self.sp, self.prompt())


def reference_emit(sp: Spec, prompt, max_new=100000):
    """Independent numpy model of the machine: parses the prompt like the hardware would and returns the tokens the model
    must emit (excluding the final HALT)."""
    W = np.zeros((sp.rows, sp.ncols), dtype=np.int64); a = np.zeros(sp.nreg, dtype=np.int64)
    succ, mask, shift = {}, {}, {}
    for t in prompt:
        if sp.is_cell(t): s, c, w = sp.cell_dec(t); W[s, c] = w
        elif sp.is_act(t): a[sp.act_reg(t)] = sp.act_val(t)
        else:
            d = sp.prog_dec(t)
            if d is None: continue
            if d[0] in ("PSR", "PSA"): succ[sp.succ_slot(t)] = sp.succ_word(t)
            elif d[0] == "PM": mask[(d[1], d[2])] = d[3]
            elif d[0] == "PSH": shift[d[1]] = d[2]
    cur, out = prompt[-1], []
    while len(out) < max_new:
        slot = sp.slot(cur)
        if sp.is_read(cur):
            j = slot
            y = 0
            for f in range(sp.fan):
                c = mask.get((f, j), sp.ncols)
                if c != sp.ncols:
                    t = sp.col_tile(c); y += int(a[t * sp.rows:(t + 1) * sp.rows] @ W[:, c])
            v = int(np.clip(y >> shift.get(j, 0), 0, sp.amax)) if (j in shift or not sp.smax) else 0
            nxt = succ.get(slot, 0) + v
        elif sp.is_act(cur): nxt = succ.get(slot, sp.HALT)
        else: nxt = sp.NOP
        if nxt == sp.HALT: break
        out.append(nxt); cur = nxt
        if sp.is_act(nxt): a[sp.act_reg(nxt)] = sp.act_val(nxt)
    return out
