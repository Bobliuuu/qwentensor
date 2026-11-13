```bash
pip install -e ".[dev]"                 # torch, numpy, qwenTPU (from GitHub), pytest, scikit-learn
OMP_NUM_THREADS=1 python -m pytest -q tests   # 83 tests, about 3 min
python examples/digits_mlp.py           # float vs int accuracy (also deep_mlp.py, cnn.py, residual_mlp.py)
python examples/run_digits_mlp.py 20    # qt.compile + run 20 digits on the circuit simulator, assert machine == reference
                                        # (also run_deep_mlp.py, run_residual.py, run_cnn.py: smaller variants that fit the machine)
```

```python
import qwentensor as qt

# model: a trained float nn.Module; Xtr, ytr: training samples and labels (N, *input_shape)
prog = qt.compile(model, Xtr, input_shape=(15,), labels=ytr,
                  qat=dict(epochs=60, lr=0.01), backend="sim")   # "qwen" = real Qwen3.5 (CUDA + QWEN_DIR); see scripts/gpu_run.py
print(prog.explain())                  # neurons, ops, registers, columns, weight cells, prompt length
y = prog.run(Xte)                      # integer output vectors decoded from the emitted tokens;
                                       # asserts equality with reference.run (qt.MachineMismatch otherwise)
pred = y.argmax(1)
prog.accuracy(Xte, yte, on="float")    # also on="int" (golden) and on="machine"
prog.save("net.json")                  # qt.load("net.json")
```

`qt.compile(model, calib_data, input_shape, profile="v2.1", backend="sim"|"qwen", labels=None, qat=dict(...), ab=5, wb=5, bias_rows=1, rows=16, verify=True)` returns a `Program` with
`run`, `predict`, `check`, `reference`, `accuracy(on="machine"|"int"|"float")`, `cost`, `explain`, `save`; `qt.load` reads it back.
Errors: `qt.UnsupportedOpError` names the op; `qt.DoesNotFit` gives the neuron count, a per-layer breakdown, the limits and what to change; `qt.MachineMismatch` means the machine output differs from `reference.run` (an executor bug).
`DoesNotFit` also checks the 126-column limit: every register tile a neuron reads costs one column, so a 64-input layer needs about 5 columns per neuron.
Step-by-step (what `compile` does): `frontend.from_torch` -> `quant.quantize(fg, X_calib, QConfig("v2.1", bias_rows=4))` -> optional `quant.qat` ->
`machine.sched.compile_graph` -> `Program.prompt()`.
