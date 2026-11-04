"""Hook the machine circuit into qwentpu's merge/driver (which read `qwentpu.circuit.build` / `.VOCAB`): no qwentpu file is touched."""
from qwentpu import circuit as C

from .circuit import build


def install(sp):
    """Make qwentpu.merge.build_merged / qwentpu.driver.TpuSession build and feed THIS machine."""
    C.build = lambda: build(sp)
    C.VOCAB = list(range(sp.nvocab))
    C.CODE = sp.code_bits
    return sp
