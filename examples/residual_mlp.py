"""Tiny residual MLP: 64 -> 32 (ReLU), one residual block h = relu(h + fc2(relu(fc1(h)))), 32 -> 10.
The add is free: it becomes identity rows of the matvec that consumes concat[relu(fc1(h)), h]."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from common import digits, train_float, report
from qwentensor.quant import QConfig


class ResMLP(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.inp = nn.Linear(64, w); self.fc1 = nn.Linear(w, w); self.fc2 = nn.Linear(w, w); self.out = nn.Linear(w, 10)

    def forward(self, x):
        h = torch.relu(self.inp(x))
        h = torch.relu(h + self.fc2(torch.relu(self.fc1(h))))
        return self.out(h)


Xtr, Xte, ytr, yte = digits("raw64")
torch.manual_seed(0)
model = ResMLP()
train_float(model, Xtr, ytr, epochs=150)
if __name__ == "__main__":
    q = {"QAT from PTQ (40 ep)": dict(epochs=40, lr=0.01)}
    report("residual MLP", model, (64,), Xtr, ytr, Xte, yte, {
        "v2.0 no-shift, bias_rows=1": (QConfig("v2.0"), q),
        "v2.1 shift, bias_rows=1": (QConfig("v2.1"), q),
        "v2.1 shift, bias_rows=4": (QConfig("v2.1", bias_rows=4), q),
    })
