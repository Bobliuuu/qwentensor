"""Stage 1: program memory.  The successor schedule is prompt data (PROG tokens), not compiled in.
ONE compiled circuit (one `IRMachine`) runs different topologies / orders / slot numberings."""
import numpy as np

from qwentensor.machine.asm import Op, Program
from qwentensor.machine.run import IRMachine
from qwentensor.machine.spec import Spec


def check(m, prog):
    p = prog.prompt(); em = prog.expected()
    pred = m.run(p, em)
    assert pred == em + [m.sp.HALT], (pred[:10], em[:10])
    return p, em


def two_layer(sp, rng, slots, order_hidden, skip=False):
    """regs: 0,1 inputs; 2,3,4 hidden; 5,6 outputs; 7 = constant 1.  cols 0..2 hidden neurons, 3,4 output neurons."""
    W = {}
    for c in range(3):
        for s in (0, 1, 7): W[(s, c)] = int(rng.integers(-1, 3))
    for c in (3, 4):
        for s in (2, 3, 4, 7) + ((0,) if skip else ()): W[(s, c)] = int(rng.integers(-1, 3))
    ops = {}
    hid = {c: Op([c], 0, 2 + c, f"h{c}") for c in range(3)}
    out = {3: Op([3], 0, 5, "o0"), 4: Op([4], 0, 6, "o1")}
    seq = [hid[c] for c in order_hidden] + [out[3], out[4]]
    for slot, o in zip(slots, seq): ops[slot] = o
    prog = Program(sp, W, ops, list(slots), {0: int(rng.integers(0, 4)), 1: int(rng.integers(0, 4)), 7: 1}, entry=7)
    return prog


def test_program_memory_many_programs_one_circuit():
    sp = Spec(rows=8, cols=(8,), ab=3, wb=3, nops=8)
    m = IRMachine(sp)
    print(m.stats())
    rng = np.random.default_rng(0)
    seen = set()
    for slots, order, skip in [([0, 1, 2, 3, 4], [0, 1, 2], False), ([6, 2, 5, 0, 7], [2, 0, 1], False),
                               ([3, 1, 4, 7, 0], [1, 2, 0], True)]:
        for _ in range(2):
            prog = two_layer(sp, rng, slots, order, skip)
            p, em = check(m, prog)
            seen.add(tuple(em)); assert len(em) == 10
    assert len(seen) >= 4
    # prompt = weights + program image + register init only: no READ / emitted-class token anywhere
    assert all(not sp.is_read(t) for t in p)


def test_different_length_programs():
    """A 1-op, 3-op and 5-op program on the same circuit (depth is data)."""
    sp = Spec(rows=8, cols=(8,), ab=3, wb=3, nops=8)
    m = IRMachine(sp)
    rng = np.random.default_rng(3)
    p1 = Program(sp, {(0, 0): 2, (7, 0): 1}, {5: Op([0], 0, 3)}, [5], {0: 3, 7: 1}, entry=7)
    assert len(check(m, p1)[1]) == 2
    prog = two_layer(sp, rng, [0, 1, 2, 3, 4], [0, 1, 2]); assert len(check(m, prog)[1]) == 10
