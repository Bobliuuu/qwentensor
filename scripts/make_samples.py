"""Write qwentensor/machine/data/digits_samples.npz: encoded held-out digits + labels (needs sklearn/torch; run locally once)."""
import sys, os
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "examples"), os.path.join(os.path.dirname(__file__), "..")]
import numpy as np
from common import digits
from qwentensor import reference as R
from qwentensor.ir import Graph

Xtr, Xte, ytr, yte = digits("pool15")
for prof in ("v21", "v20"):
    g = Graph.load(f"qwentensor/machine/data/digits_{prof}.json")
    xq = R.encode_input(g, Xte)
    np.savez(f"qwentensor/machine/data/digits_samples_{prof}.npz", xq=xq, y=yte, ref=R.run(g, xq))
    print(prof, xq.shape, "int acc", (R.run(g, xq).argmax(1) == yte).mean())
