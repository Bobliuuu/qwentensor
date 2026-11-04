"""Stage 2: K-tiling.  A neuron with K > rows-1 inputs spans several tiles: one column per tile, summed EXACTLY in the adder
tree (counts of the accumulator heads of all masked columns are added before any shift / relu / clamp)."""
import numpy as np

from qwentensor.machine.asm import Op, Program
from qwentensor.machine.run import IRMachine
from qwentensor.machine.spec import Spec
from test_machine_prog import check


def test_k_tiled_neurons():
    sp = Spec(rows=8, cols=(3, 3, 2), ab=3, wb=3, nops=8, fan=2)
    m = IRMachine(sp); print(m.stats())
    rng = np.random.default_rng(5)
    W = {}
    # layer 1: hidden neuron n (n<3) = column n in tile 0 (rows 0..6 = inputs 0..6, bias row 7) + column 3+n in tile 1 (inputs 7..13)
    for n in range(3):
        for s in range(8): W[(s, n)] = int(rng.integers(-2, 3)) if s < 7 else int(rng.integers(0, 3))
        for s in range(7): W[(s, 3 + n)] = int(rng.integers(-2, 3))
    # layer 2: output neuron o (o<2) = column 6+o in tile 2 over hidden regs 16..18, bias reg 23
    for o in range(2):
        for s in (0, 1, 2, 7): W[(s, 6 + o)] = int(rng.integers(0, 3))
    ops = {0: Op([0, 3], 0, 16), 1: Op([1, 4], 0, 17), 2: Op([2, 5], 0, 18), 3: Op([6], 0, 19), 4: Op([7], 0, 20)}
    init = {r: int(rng.integers(0, 4)) for r in list(range(7)) + list(range(8, 15))}
    init[7] = 1
    prog = Program(sp, W, ops, [0, 1, 2, 3, 4], init, entry=7)
    p, em = check(m, prog)
    # the neuron really sums across tiles: compare with plain numpy of the whole 14+1 input vector
    x = np.array([init[r] for r in range(7)] + [init[r] for r in range(8, 15)]); b = 1
    for n in range(3):
        w = np.array([W[(s, n)] for s in range(7)] + [W[(s, 3 + n)] for s in range(7)]); bias = W[(7, n)]
        y = int(np.clip(x @ w + bias, 0, sp.amax))
        assert em[2 * n + 1] == sp.act(16 + n, y), n
