"""A deeper/wider MLP on all 64 pixels: 64 -> 96 -> 64 -> 32 -> 10 (4 matvec layers, hidden widths up to 96 > the 15-wide array: needs tiling)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from common import digits, train_float, report
from qwentensor.quant import QConfig

Xtr, Xte, ytr, yte = digits("raw64")
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(64, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
train_float(model, Xtr, ytr, epochs=150)
if __name__ == "__main__":
    q = {"QAT from PTQ (40 ep)": dict(epochs=40, lr=0.01)}
    report("deep MLP 64-96-64-32-10", model, (64,), Xtr, ytr, Xte, yte, {
        "v2.0 no-shift, bias_rows=1": (QConfig("v2.0"), q),
        "v2.1 shift, bias_rows=1": (QConfig("v2.1"), q),
        "v2.1 shift, bias_rows=4": (QConfig("v2.1", bias_rows=4), q),
    })
