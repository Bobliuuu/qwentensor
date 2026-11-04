"""Write scripts/data/residual_v21.json (+ samples npz): the 15 -> 10 -> residual(10) -> 10 digits network from examples/run_residual.py,
quantized with the v2.1 profile. Run locally once (needs torch + sklearn); scripts/gpu_run.py residual reads it on the A100."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "examples"), os.path.join(HERE, "..")]
import numpy as np, torch, torch.nn as nn
import qwentensor as qt
from qwentensor import reference as R
from common import digits, train_float, torch_acc

Xtr, Xte, ytr, yte = digits("pool15")


class Res(nn.Module):
    def __init__(s, w=10):
        super().__init__(); s.i = nn.Linear(15, w); s.a = nn.Linear(w, w); s.b = nn.Linear(w, w); s.o = nn.Linear(w, 10)
    def forward(s, x):
        h = torch.relu(s.i(x)); h = torch.relu(h + s.b(torch.relu(s.a(h)))); return s.o(h)


torch.manual_seed(0); model = Res(); train_float(model, Xtr, ytr, epochs=300)
prog = qt.compile(model, Xtr, input_shape=(15,), labels=ytr, qat=dict(epochs=40, lr=0.01))
g = prog.graph
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
g.save(os.path.join(HERE, "data", "residual_v21.json"))
xq = R.encode_input(g, Xte); ref = R.run(g, xq)
np.savez(os.path.join(HERE, "data", "residual_samples_v21.npz"), xq=xq, y=yte, ref=ref)
print("float acc", torch_acc(model, Xte, yte), "int acc", (ref.argmax(1) == yte).mean())
print(prog.explain())
