"""Networks larger than one machine, run in pages (docs/STREAMING.md).
usage: python examples/run_stream.py [deep|residual|cnn] [n_circuit]      (default: deep 2)

Compiles the network with qt.compile(..., paged=True), then checks the machine against reference.run on
  * all 540 held-out digits with the numpy token golden of every page (fast), and
  * the first n_circuit digits through the CPU circuit-IR simulator (every page, every exactness condition asserted)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import qwentensor as qt
from qwentensor import reference as R
from qwentensor.machine.pages import golden_executor
import stream_models as S

which = sys.argv[1] if len(sys.argv) > 1 else "deep"
n_circuit = int(sys.argv[2]) if len(sys.argv) > 2 else 2
if which == "cnn":
    model, Xtr, Xte, ytr, yte = S.cnn(); shape, qat = (1, 8, 8), dict(epochs=20, lr=0.01)
else:
    model = S.deep() if which == "deep" else S.residual(); Xtr, Xte, ytr, yte = S.Xtr, S.Xte, S.ytr, S.yte; shape, qat = (44,), dict(epochs=40, lr=0.01)
prog = qt.compile(model, Xtr, input_shape=shape, labels=ytr, qat=qat, paged=True, max_cols=45, max_ops=60)
print(prog.explain(), "\n")
xq = R.encode_input(prog.graph, Xte); ref = R.run(prog.graph, xq)
print(f"integer accuracy (reference.run), {len(xq)} held-out digits: {(ref.argmax(1) == yte).mean():.4f}")
ex = golden_executor(); t = time.time()
out = np.array([prog.paged.run(x, ex)[0] for x in xq])
print(f"paged machine (token golden) == reference.run on {len(xq)} digits: {bool((out == ref).all())} ({time.time()-t:.0f}s)")
t = time.time(); y = prog.run(Xte[:n_circuit], check=True)
print(f"paged machine (circuit simulator) == reference.run on {n_circuit} digits: True ({time.time()-t:.0f}s, "
      f"{prog.last['pages']} pages, {prog.last['prompt_tokens']} prompt tokens in total for the last digit)")
