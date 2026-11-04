"""Shared helpers for the examples: digits data, float training, the standard float->PTQ->QAT report."""
import numpy as np
import torch
import torch.nn as nn

from qwentensor import frontend, quant, reference as R


def digits(features="pool15", seed=0):
    """sklearn 8x8 digits, same split as qwenTPU examples/mlp/train.py (30% test = 540 digits, stratified, random_state=0).
    pool15: 2x2 average pool -> 4x4 blocks, drop the always-dark corner -> 15 features in 0..1 (qwenTPU's input).
    raw64: all 64 pixels in 0..1 (as (1,8,8) images when image=True is requested by reshaping the result)."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits(); I = d.images / 16.0
    Z = I.reshape(-1, 4, 2, 4, 2).mean((2, 4)).reshape(len(I), 16)[:, 1:] if features == "pool15" else I.reshape(len(I), 64)
    return train_test_split(Z, d.target, test_size=0.3, random_state=seed, stratify=d.target)


def train_float(model, Xtr, ytr, epochs=300, lr=0.01, batch=64, seed=1, wd=0.0, shape=None):
    torch.manual_seed(seed); torch.set_num_threads(1); model.double()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    X = torch.tensor(Xtr, dtype=torch.float64); y = torch.tensor(ytr)
    if shape: X = X.reshape(len(X), *shape)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def torch_acc(model, X, y, shape=None):
    with torch.no_grad():
        X = np.asarray(X).reshape(len(X), *shape) if shape else X
        return float((model(torch.tensor(X, dtype=torch.float64)).argmax(1).numpy() == y).mean())


def report(name, model, input_shape, Xtr, ytr, Xte, yte, configs, **kw):
    """float torch acc, float-IR acc, then per config: PTQ accuracy and each QAT variant's accuracy on the held-out test set.
    configs: {label: (QConfig, {qat_label: qat_kwargs})}.  Returns a dict of results (and the quantized graphs)."""
    facc = torch_acc(model, Xte, yte, input_shape)
    fg = frontend.from_torch(model, input_shape, **kw)
    iracc = quant.float_accuracy(fg, Xte, yte)
    print(f"[{name}] float torch test acc {facc:.4f} | float IR {iracc:.4f} ({len(yte)} test samples)")
    out = {}
    for label, (cfg, qats) in configs.items():
        q = quant.quantize(fg, Xtr, cfg); p = quant.accuracy(q, Xte, yte)
        print(f"   {label:30s} PTQ                 int acc {p:.4f}   shifts {[n.shift for n in q.matvecs()]}", flush=True)
        out[label] = {"ptq": (p, q)}
        for ql, kwargs in qats.items():
            q2 = quant.qat(q, Xtr, ytr, **kwargs); a2 = quant.accuracy(q2, Xte, yte)
            print(f"   {label:30s} {ql:20s} int acc {a2:.4f}   (train {q2.meta.get('qat_train_acc', 0):.4f})", flush=True)
            out[label][ql] = (a2, q2)
    return dict(float_torch=facc, float_ir=iracc, results=out, fg=fg)
