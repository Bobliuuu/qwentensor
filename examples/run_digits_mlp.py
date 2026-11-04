"""End-to-end: PyTorch digits MLP (15-15-10) -> qt.compile -> run on the CPU circuit simulator; machine == reference on every held-out digit.
usage: OMP_NUM_THREADS=1 python examples/run_digits_mlp.py [n_test]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import qwentensor as qt
from digits_mlp import model, Xtr, Xte, ytr, yte
from common import torch_acc

n = int(sys.argv[1]) if len(sys.argv) > 1 else len(yte)
prog = qt.compile(model, Xtr, input_shape=(15,), labels=ytr, qat=dict(epochs=60, lr=0.01), backend="sim")
print(prog.explain()); print()
t = time.time(); y = prog.run(Xte[:n], check=True); dt = time.time() - t
ref = prog.reference(Xte[:n])
print(f"machine == reference.run on {n}/{n} digits: {bool((y == ref).all())}   ({dt:.1f}s, {dt/n:.2f}s/digit, prompt {prog.last['prompt_tokens']} tokens)")
print(f"accuracy on {n} held-out digits: float torch {torch_acc(model, Xte[:n], yte[:n]):.4f} | int (reference) {float((ref.argmax(1)==yte[:n]).mean()):.4f} | machine {float((y.argmax(1)==yte[:n]).mean()):.4f}")
