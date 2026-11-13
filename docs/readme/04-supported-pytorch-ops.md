Authoritative table: [docs/IR.md](docs/IR.md) section 3.

| supported | rejected with `UnsupportedOpError` (and how it could be added, see DESIGN.md section 10) |
|---|---|
| `Linear`, `ReLU`, `Conv2d` (im2col, groups=1, dilation=1, zero padding, any stride), residual `Add`, `AvgPool2d` / `AdaptiveAvgPool2d`, `Flatten` / `view` / `reshape`, `Embedding` on the integer input, `Identity` / `Dropout` (eval), trailing `argmax` | `Softmax` / `log_softmax` (a trailing one before argmax is dropped), `LayerNorm`, `BatchNorm` (fold it first; not automatic yet), `GELU` / `SiLU` / `Sigmoid` / `Tanh`, `MaxPool2d`, attention, data x data products, recurrent layers, control flow |
