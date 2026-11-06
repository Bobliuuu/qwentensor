"""Write qwentensor/machine/data/wide_{deep,residual,cnn}.json (the ORIGINAL examples: 64-96-64-32-10 deep MLP, 64-32-[32+res]-10 residual MLP,
2-conv + AvgPool + 512-wide Linear CNN, quantized) and wide_samples_*.npz (encoded held-out digits, labels, reference.run outputs) for
scripts/gpu_run.py wide.  Needs sklearn + torch; run locally once."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "examples"), os.environ.get("QWENTPU_SRC", os.path.expanduser("~/qwentpu_src/qwentpu"))]
import numpy as np
import qwentensor as qt
from qwentensor import reference as R
import deep_mlp, residual_mlp, cnn

DATA = os.path.join(HERE, "..", "qwentensor", "machine", "data")
nets = (("deep", deep_mlp.model, deep_mlp.Xtr, deep_mlp.Xte, deep_mlp.ytr, deep_mlp.yte, (64,), dict(epochs=40, lr=0.01)),
        ("residual", residual_mlp.model, residual_mlp.Xtr, residual_mlp.Xte, residual_mlp.ytr, residual_mlp.yte, (64,), dict(epochs=40, lr=0.01)),
        ("cnn", cnn.model, cnn.Xtr, cnn.Xte, cnn.ytr, cnn.yte, (1, 8, 8), dict(epochs=20, lr=0.01)))
for name, model, xtr, xte, ytr, yte, shape, qat in nets:
    if len(sys.argv) > 1 and name not in sys.argv[1:]: continue
    t = time.time()
    prog = qt.compile(model, xtr, input_shape=shape, labels=ytr, qat=qat, paged=True, max_cols=45, max_ops=60)
    xq = R.encode_input(prog.graph, xte); ref = R.run(prog.graph, xq)
    prog.graph.save(os.path.join(DATA, f"wide_{name}.json"))
    np.savez(os.path.join(DATA, f"wide_samples_{name}.npz"), xq=xq, y=yte, ref=ref)
    print(name, "neurons", sum(len(p.neurons) for p in prog.paged.pages), "pages", len(prog.paged.pages), "machine columns", prog.spec.ncols, "tiles", prog.spec.tiles,
          "fan", prog.spec.fan, "groups", prog.spec.ngroups, "int acc", float((ref.argmax(1) == yte).mean()), "shifts", [n.shift for n in prog.graph.matvecs()],
          f"{time.time()-t:.0f}s", flush=True)
