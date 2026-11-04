"""Residual MLP (64 -> 32 -> residual block -> 10 = 106 neurons) through qt.compile on the simulator.  usage: python examples/run_residual.py [n_test]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import qwentensor as qt
from common import torch_acc
import torch, torch.nn as nn
from common import digits, train_float
from residual_mlp import model as big, ResMLP, Xtr as Xtr64
Xtr, Xte, ytr, yte = digits("pool15")

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
try:
    qt.compile(big, Xtr64, input_shape=(64,))
except qt.DoesNotFit as e:
    print("residual 64-32-res32-10 -> DoesNotFit:\n ", e, "\n")


class Res(nn.Module):       # 15 -> 10, one residual block (width 10), 10 -> 10
    def __init__(s, w=10):
        super().__init__(); s.i = nn.Linear(15, w); s.a = nn.Linear(w, w); s.b = nn.Linear(w, w); s.o = nn.Linear(w, 10)
    def forward(s, x):
        h = torch.relu(s.i(x)); h = torch.relu(h + s.b(torch.relu(s.a(h)))); return s.o(h)


torch.manual_seed(0); model = Res(); train_float(model, Xtr, ytr, epochs=300)
try:
    prog = qt.compile(model, Xtr, input_shape=(15,), labels=ytr, qat=dict(epochs=40, lr=0.01))
except qt.DoesNotFit as e:
    print("DoesNotFit:", e); sys.exit(1)
print(prog.explain(), "\n")
t = time.time(); y = prog.run(Xte[:n]); dt = time.time() - t
ref = prog.reference(Xte[:n])
print(f"machine == reference.run on {n}/{n} digits: {bool((y == ref).all())} ({dt:.0f}s, prompt {prog.last['prompt_tokens']} tokens)")
print(f"first {n} held-out: float torch {torch_acc(model, Xte[:n], yte[:n]):.3f} | int {float((ref.argmax(1)==yte[:n]).mean()):.3f} | machine {float((y.argmax(1)==yte[:n]).mean()):.3f}")
print(f"all {len(yte)} held-out: float torch {torch_acc(model, Xte, yte):.4f} | int {prog.accuracy(Xte, yte, 'int'):.4f}")
