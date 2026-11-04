"""PyTorch deep MLP 64-96-64-32-10 through qt.compile.  Needs 202 neurons (> 128 op slots): prints the DoesNotFit message, then runs a
smaller deep MLP (15-24-16-10 on 2x2-pooled digits, 50 neurons) end to end if it fits.  usage: python examples/run_deep_mlp.py [n_test]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn as nn
import qwentensor as qt
from common import digits, train_float, torch_acc
from deep_mlp import model as big

Xtr64, _, _, _ = digits("raw64")
Xtr, Xte, ytr, yte = digits("pool15")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
try:
    qt.compile(big, Xtr64, input_shape=(64,), backend="sim")
except qt.DoesNotFit as e:
    print("64-96-64-32-10 -> DoesNotFit:\n ", e, "\n")
torch.manual_seed(0)
small = nn.Sequential(nn.Linear(15, 24), nn.ReLU(), nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, 10))
train_float(small, Xtr, ytr, epochs=150)
prog = qt.compile(small, Xtr, input_shape=(15,), labels=ytr, qat=dict(epochs=30, lr=0.01))
print(prog.explain(), "\n")
t = time.time(); y = prog.run(Xte[:n]); dt = time.time() - t
ref = prog.reference(Xte[:n])
print(f"machine == reference.run on {n}/{n} digits: {bool((y == ref).all())} ({dt:.0f}s, prompt {prog.last['prompt_tokens']} tokens)")
print(f"first {n} held-out: float torch {torch_acc(small, Xte[:n], yte[:n]):.3f} | int {float((ref.argmax(1)==yte[:n]).mean()):.3f} | machine {float((y.argmax(1)==yte[:n]).mean()):.3f}")
print(f"all {len(yte)} held-out: float torch {torch_acc(small, Xte, yte):.4f} | int {prog.accuracy(Xte, yte, 'int'):.4f}")
