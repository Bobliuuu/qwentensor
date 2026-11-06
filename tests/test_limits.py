"""Boundary tests behind docs/LIMITS.md: each asserts where a machine bound sits today.  When a circuit change moves a bound
(wide layers, vectorisation), the matching test fails on purpose: update the test and docs/LIMITS.md together."""
import numpy as np
import pytest

from qwentensor.machine.asm import Op, Program
from qwentensor.machine.circuit import build
from qwentensor.machine.run import IRMachine
from qwentensor.machine.spec import Spec


def sp(rows=16, ncols=8, nops=8, fan=1, smax=5, ab=5, wb=5):
    return Spec(rows=rows, cols=(ncols,), nops=nops, fan=fan, smax=smax, ab=ab, wb=wb)


def test_columns_and_ops_ceiling_is_the_head_dimension():
    """HD = 128: a mask head stores one word per op slot (<= 128 slots) with 2 reference columns + one column bit each (<= 126)."""
    build(sp(ncols=126, nops=126))
    build(sp(ncols=126, nops=128))
    with pytest.raises(AssertionError): build(sp(ncols=127, nops=128))
    with pytest.raises(AssertionError): build(sp(ncols=126, nops=129))


def test_requant_window_bounds_fan_times_rows_by_shift():
    """The carry staircase of the requantiser must fit the 256-wide gate window: fan*rows <= 48 for shifts >= 5."""
    build(sp(rows=16, ncols=3, fan=3))
    with pytest.raises(ValueError, match="gate window"): build(sp(rows=16, ncols=4, fan=4))


def test_bf16_counts_are_exact_only_up_to_256():
    import torch
    assert float(torch.tensor(256.0).to(torch.bfloat16)) == 256.0 and float(torch.tensor(257.0).to(torch.bfloat16)) == 256.0


def _stress(rows, fan, smax, trials=6):
    """All activations at amax, weights near wmax with random cleared bits: the digit mux is pushed to ab*fan*rows."""
    s = Spec(rows=rows, cols=(1,) * fan + (0,), ab=5, wb=5, nops=2, fan=fan, smax=smax)
    m = IRMachine(s); rng = np.random.default_rng(1); ok = 0
    for tr in range(trials):
        N = rows * fan; W = {}
        for i in range(N):
            w = s.whi
            for b in range(s.wb - 1):
                if rng.random() < (0.0, 0.05, 0.1, 0.2)[tr % 4]: w &= ~(1 << b)
            if w: W[(i % rows, i // rows)] = int(w)
        prog = Program(s, W, {0: Op(list(range(fan)), smax, N)}, [0], {i: s.amax for i in range(N)}, entry=rows - 1)
        p = prog.prompt(); em = prog.expected()
        try: ok += m.run(p, em) == em + [s.HALT]
        except AssertionError: pass
    return ok, trials


def test_fan_times_rows_64_is_exact_and_80_is_not():
    """Builds are not exactness: fan*rows = 80 at smax 4 builds, but digit-mux values up to 4*80 = 320 leave bf16's exact integers
    (<= 256 for odd values) and the circuit evaluator rejects some inputs.  64 = 256 / 4 is exact."""
    ok, n = _stress(16, 4, 4); assert ok == n
    ok, n = _stress(16, 5, 4, trials=8); assert ok < n


def test_activation_bits_ceiling_is_nine():
    build(sp(ab=9, wb=9))
    with pytest.raises(AssertionError): build(sp(ab=10, wb=10))
