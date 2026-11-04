"""qwenTPU's digits classifier (15 -> 15 -> 10, 5-bit, 2x2-pooled features) rebuilt through the PyTorch frontend.

qwenTPU's hand-written pipeline (examples/mlp/train.py): float 91.1%, integer (v2.0, no shift, trained with STE from scratch) 87.6%
on the same 540 held-out digits.  Here: train a torch MLP, trace it, post-training-quantize, optionally fine-tune with STE."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from common import digits, train_float, report
from qwentensor.quant import QConfig

Xtr, Xte, ytr, yte = digits("pool15")
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(15, 15), nn.ReLU(), nn.Linear(15, 10))
train_float(model, Xtr, ytr, epochs=300)
if __name__ == "__main__":
    ptq = {"QAT from PTQ (60 ep)": dict(epochs=60, lr=0.01)}
    scratch = {"QAT from PTQ (60 ep)": dict(epochs=60, lr=0.01),
               "qwenTPU-style QAT from scratch": dict(epochs=250, lr=0.02, seeds=4, init="random")}
    report("digits 15-15-10", model, (15,), Xtr, ytr, Xte, yte, {
        "v2.0 no-shift, input 0..12": (QConfig("v2.0", input_max=12), scratch),
        "v2.0 no-shift, input 0..31": (QConfig("v2.0"), ptq),
        "v2.1 shift,    input 0..31": (QConfig("v2.1"), ptq),
        "v2.1 shift, bias_rows=4": (QConfig("v2.1", bias_rows=4), ptq),
    })
