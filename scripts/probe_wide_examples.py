"""Do the original 64-input examples fit when paged?  (docs/STREAMING.md 3.2)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "examples"), os.environ.get("QWENTPU_SRC", os.path.expanduser("~/qwentpu_src/qwentpu"))]
import qwentensor as qt
from common import digits

Xtr, _, ytr, _ = digits("raw64")
for name in ("deep_mlp", "residual_mlp"):
    m = __import__(name).model
    for kw in (dict(), dict(bias_rows=4)):
        try:
            p = qt.compile(m, Xtr, input_shape=(64,), labels=ytr, qat=dict(epochs=5, lr=0.01), paged=True, max_cols=45, max_ops=60, **kw)
            print(name, kw, "FITS", p.spec)
        except qt.DoesNotFit as e:
            print(name, kw, "DoesNotFit:", str(e)[:260])
