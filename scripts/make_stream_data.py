"""Write qwentensor/machine/data/stream_{deep,residual,cnn}.json (quantized integer graphs) and stream_samples_*.npz (encoded held-out digits,
labels, reference.run outputs) for scripts/gpu_run.py stream.  Needs sklearn + torch; run locally once."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "examples"), *[p for p in [os.environ.get("QWENTPU_SRC")] if p]]
import numpy as np
import qwentensor as qt
from qwentensor import reference as R
import stream_models as S

DATA = os.path.join(HERE, "..", "qwentensor", "machine", "data")
cm, cxtr, cxte, cytr, cyte = S.cnn()
nets = (("deep", S.deep(), S.Xtr, S.Xte, S.ytr, S.yte, (44,), dict(epochs=40, lr=0.01)),
        ("residual", S.residual(), S.Xtr, S.Xte, S.ytr, S.yte, (44,), dict(epochs=40, lr=0.01)),
        ("cnn", cm, cxtr, cxte, cytr, cyte, (1, 8, 8), dict(epochs=20, lr=0.01)))
for name, model, xtr, xte, ytr, yte, shape, qat in nets:
    prog = qt.compile(model, xtr, input_shape=shape, labels=ytr, qat=qat, paged=True, max_cols=45, max_ops=60)
    xq = R.encode_input(prog.graph, xte); ref = R.run(prog.graph, xq)
    prog.graph.save(os.path.join(DATA, f"stream_{name}.json"))
    np.savez(os.path.join(DATA, f"stream_samples_{name}.npz"), xq=xq, y=yte, ref=ref)
    print(name, "neurons", sum(len(p.neurons) for p in prog.paged.pages), "pages", len(prog.paged.pages), "machine columns", prog.spec.ncols,
          "int acc", float((ref.argmax(1) == yte).mean()), "shifts", [n.shift for n in prog.graph.matvecs()])
