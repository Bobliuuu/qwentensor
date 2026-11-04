"""CNN through qt.compile.  The frontend example (8x8 image, conv 1->4, conv 4->8, pool, fc) needs 778 neurons (> 128 op slots) -> DoesNotFit
message; then a small CNN on the 2x2-pooled 4x4 digit (conv3x3 pad1 1->2, ReLU, fc 32->10: 42 neurons) that fits and runs on the simulator.
usage: python examples/run_cnn.py [n_test]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn as nn
import qwentensor as qt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from common import digits, train_float, torch_acc
from cnn import model as big

Xtr64, _, _, _ = digits("raw64")
try:
    qt.compile(big, Xtr64, input_shape=(1, 8, 8), backend="sim")
except qt.DoesNotFit as e:
    print("CNN 2conv+pool+fc on 8x8 -> DoesNotFit:\n ", e, "\n")

d = load_digits(); Z = (d.images / 16.0).reshape(-1, 4, 2, 4, 2).mean((2, 4)).reshape(-1, 16)      # 4x4 images in 0..1
Xtr, Xte, ytr, yte = train_test_split(Z, d.target, test_size=0.3, random_state=0, stratify=d.target)


class Tiny(nn.Module):
    def __init__(s):
        super().__init__(); s.c = nn.Conv2d(1, 2, 3, padding=1); s.f = nn.Linear(2 * 4 * 4, 10)
    def forward(s, x): return s.f(torch.flatten(torch.relu(s.c(x)), 1))


torch.manual_seed(0); tiny = Tiny()
train_float(tiny, Xtr, ytr, epochs=300, shape=(1, 4, 4))
try:
    prog = qt.compile(tiny, Xtr, input_shape=(1, 4, 4), labels=ytr, qat=dict(epochs=40, lr=0.01))
except qt.DoesNotFit as e:
    print("tiny CNN -> DoesNotFit:", e); sys.exit(1)
print(prog.explain(), "\n")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
t = time.time(); y = prog.run(Xte[:n]); dt = time.time() - t
ref = prog.reference(Xte[:n])
print(f"machine == reference.run on {n}/{n} digits: {bool((y == ref).all())} ({dt:.0f}s)")
print(f"all {len(yte)} held-out: float torch {torch_acc(tiny, Xte, yte, (1,4,4)):.4f} | int {prog.accuracy(Xte, yte, 'int'):.4f}")
