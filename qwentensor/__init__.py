"""qwentensor: PyTorch networks compiled to a prompt that Qwen3.5 executes.   prog = qt.compile(model, calib, input_shape=...); y = prog.run(x)"""
from .api import compile, load, Program, DoesNotFit, MachineMismatch, UnsupportedOpError  # noqa: F401
from .quant import QConfig                                                             # noqa: F401
