"""Write qwentensor/machine/data/stream_{deep,residual}.json (quantized integer graphs) and stream_samples_*.npz (encoded held-out digits,
labels, reference.run outputs) for scripts/gpu_run.py stream.  Needs sklearn + torch; run locally once."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "examples"), os.environ.get("QWENTPU_SRC", os.path.expanduser("~/qwentpu_src/qwentpu"))]
import numpy as np
import qwentensor as qt
from qwentensor import reference as R
import stream_models as S

DATA = os.path.join(HERE, "..", "qwentensor", "machine", "data")
for name, model in (("deep", S.deep()), ("residual", S.residual())):
    prog = qt.compile(model, S.Xtr, input_shape=(44,), labels=S.ytr, qat=dict(epochs=40, lr=0.01), paged=True, max_cols=45, max_ops=60)
    xq = R.encode_input(prog.graph, S.Xte); ref = R.run(prog.graph, xq)
    prog.graph.save(os.path.join(DATA, f"stream_{name}.json"))
    np.savez(os.path.join(DATA, f"stream_samples_{name}.npz"), xq=xq, y=S.yte, ref=ref)
    print(name, "neurons", len(prog.paged.pages) and sum(len(p.neurons) for p in prog.paged.pages), "pages", len(prog.paged.pages),
          "int acc", float((ref.argmax(1) == S.yte).mean()), "shifts", [n.shift for n in prog.graph.matvecs()])
