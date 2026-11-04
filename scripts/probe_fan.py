"""Probe which (rows, fan, bits, smax) machine circuits build (build only; no evaluation)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.environ.get("QWENTPU_SRC", os.path.expanduser("~/qwentpu_src/qwentpu"))]
from qwentensor.machine.spec import Spec
from qwentensor.machine.circuit import build

cases = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1:]] or [(16, 3, 5, 5, 4), (16, 4, 5, 5, 4), (16, 5, 5, 5, 4), (8, 6, 5, 5, 4), (16, 4, 4, 4, 4), (16, 6, 4, 4, 4)]
for rows, fan, ab, wb, smax in cases:
    sp = Spec(rows=rows, cols=(4,) * fan, ab=ab, wb=wb, nops=8, fan=fan, smax=smax)
    try:
        t = time.time(); g, d = build(sp)
        print(rows, fan, ab, wb, smax, "ok layers", g.n_layers(), f"{time.time()-t:.1f}s", flush=True)
    except Exception as e:
        print(rows, fan, ab, wb, smax, "FAIL", type(e).__name__, str(e)[:140], flush=True)
