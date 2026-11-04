#!/usr/bin/env python3
"""Real numbers for the clips: trains the examples' float nets, runs the frontend PTQ/QAT, writes data/study.json + data/digits_graph.json.
Mirrors examples/digits_mlp.py and examples/deep_mlp.py (same seeds/epochs); nothing is hard-coded. ~2-5 min on CPU."""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "examples")]
import numpy as np, torch, torch.nn as nn
from common import digits, train_float, torch_acc
from qwentensor import frontend, quant, reference as R
from qwentensor.quant import QConfig
out = {}; t0 = time.time()
# ---- digits 15-15-10
Xtr, Xte, ytr, yte = digits("pool15"); torch.manual_seed(0)
model = nn.Sequential(nn.Linear(15, 15), nn.ReLU(), nn.Linear(15, 10)); train_float(model, Xtr, ytr, epochs=300)
facc = torch_acc(model, Xte, yte, (15,)); fg = frontend.from_torch(model, (15,))
d = {"float": facc, "n_test": len(yte)}
for lab, cfg in [("v20_ptq", QConfig("v2.0", input_max=12)), ("v20_ptq_in31", QConfig("v2.0")), ("v21_ptq", QConfig("v2.1")), ("v21_b4_ptq", QConfig("v2.1", bias_rows=4))]:
    q = quant.quantize(fg, Xtr, cfg); d[lab] = quant.accuracy(q, Xte, yte); d[lab + "_shifts"] = [n.shift for n in q.matvecs()]
    d[lab + "_bias_clipped"] = [r.get("b_clipped") for r in q.meta["report"]]
    if lab == "v21_ptq":
        qa = quant.qat(q, Xtr, ytr, epochs=60, lr=0.01); d["v21_qat"] = quant.accuracy(qa, Xte, yte); qa.save(os.path.join(HERE, "data", "digits_graph.json"))
        d["float_w1"] = model[0].weight.detach().numpy().round(3).tolist(); d["float_in_scale"] = None
        d["ptq_w1q"] = q.nodes[1].W.T.tolist() if q.nodes[1].W.shape == (15, 15) else None
    if lab == "v21_b4_ptq":
        qa = quant.qat(q, Xtr, ytr, epochs=60, lr=0.01); d["v21_b4_qat"] = quant.accuracy(qa, Xte, yte)
    print(lab, d[lab], time.time() - t0, flush=True)
out["digits"] = d
# ---- deep MLP (PTQ only: v2.0 vs v2.1)
Xtr, Xte, ytr, yte = digits("raw64"); torch.manual_seed(0)
m2 = nn.Sequential(nn.Linear(64, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)); train_float(m2, Xtr, ytr, epochs=150)
fg2 = frontend.from_torch(m2, (64,)); e = {"float": torch_acc(m2, Xte, yte, (64,)), "n_test": len(yte)}
for lab, cfg in [("v20_ptq", QConfig("v2.0")), ("v21_ptq", QConfig("v2.1"))]:
    q = quant.quantize(fg2, Xtr, cfg); e[lab] = quant.accuracy(q, Xte, yte); e[lab + "_shifts"] = [n.shift for n in q.matvecs()]; print("deep", lab, e[lab], flush=True)
out["deep"] = e
json.dump(out, open(os.path.join(HERE, "data", "study.json"), "w"), indent=1); print("done", time.time() - t0)
