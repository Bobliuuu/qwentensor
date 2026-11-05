"""User-facing path:   prog = qt.compile(torch_model, calib_data, input_shape=..., backend="sim"); y = prog.run(x)

Chain: frontend.from_torch -> quant.quantize (-> quant.qat) -> machine.sched.compile_graph -> Program.prompt()
       -> host loop (argmax; append) on the machine -> Compiled.decode -> compare with reference.run (bit-exact).
backend="sim"  : CPU circuit-IR simulator (exact float64 evaluator of the machine circuit).
backend="qwen" : the real merged Qwen3.5 (needs CUDA and QWEN_DIR; see scripts/gpu_run.py).
"""
import json
import os

import numpy as np

from . import frontend, quant, reference as R
from .ir import Graph
from .frontend import UnsupportedOpError            # noqa: F401  (re-exported)
from .machine.sched import compile_graph, DoesNotFit as _SchedDoesNotFit
from .machine import pages as _pages

MAX_OPS = 128
MAX_COLS = 126          # circuit.mem_head: one-hot column mask in <= 126 value bits


class DoesNotFit(_SchedDoesNotFit):
    """The graph needs more machine than exists.  Message states the needed size and what to change."""


class MachineMismatch(AssertionError):
    """The machine's decoded output differs from reference.run (this would be an executor bug)."""


def _describe_graph(g):
    rows = []
    for n in g.matvecs():
        rows.append(f"{n.name}: {n.groups * n.W.shape[1]} neurons (W {tuple(n.W.shape)} x groups {n.groups})")
    return "; ".join(rows)


def _fit_error(g, e):
    neurons = sum(n.groups * n.W.shape[1] for n in g.matvecs())
    need = f"graph needs {neurons} neurons (one machine op each; limits: {MAX_OPS} ops, {MAX_COLS} physical columns) -> [{_describe_graph(g)}]. Machine said: {e}."
    tips = ("To fit: shrink hidden widths, use fewer layers, make convs smaller (every output position of a conv is a separate "
            "neuron: 8x8 image, 3x3 conv, 4 channels = 256 neurons; use stride/pooling first or a Linear on a downsampled input), "
            "or widths so each neuron touches fewer register tiles (every tile a neuron reads costs one column). A network with more neurons than one machine "
            "can run in pages: qt.compile(..., paged=True) (docs/STREAMING.md); a neuron that reads more than about 45 registers still does not fit.")
    return DoesNotFit(need + " " + tips)


def _to_graph(model, calib, input_shape, profile, ab, wb, bias_rows, qat, labels, input_posneg, input_kind, input_max):
    """-> (float graph or None, integer machine graph)"""
    if isinstance(model, Graph):
        if model.profile != "float": return None, model.validate(machine=True)
        return model, _quantize(model, calib, profile, ab, wb, bias_rows, qat, labels, input_max)
    import torch.nn as nn
    if not isinstance(model, nn.Module):
        raise TypeError(f"qt.compile: model must be a torch.nn.Module or ir.Graph, got {type(model).__name__}")
    if input_shape is None:
        raise ValueError("qt.compile: input_shape is required for a torch model, e.g. input_shape=(1, 8, 8) (per-sample, no batch dim)")
    fg = frontend.from_torch(model, tuple(input_shape), input_kind=input_kind, input_posneg=input_posneg)
    return fg, _quantize(fg, calib, profile, ab, wb, bias_rows, qat, labels, input_max)


def _quantize(fg, calib, profile, ab, wb, bias_rows, qat, labels, input_max):
    if calib is None: raise ValueError("qt.compile: calib_data (training or representative samples, shape (N, *input_shape)) is required")
    calib = np.asarray(calib)
    cfg = quant.QConfig(profile=profile, ab=ab, wb=wb, bias_rows=bias_rows, input_max=input_max)
    q = quant.quantize(fg, calib, cfg)
    if qat:
        if labels is None: raise ValueError("qat=... needs labels=(N,) class labels for the calibration data")
        kw = dict(qat) if isinstance(qat, dict) else {}
        q = quant.qat(q, calib, np.asarray(labels), **kw)
    return q.validate(machine=True)


class Program:
    """A compiled network: integer graph + machine program.  `run` executes on the chosen backend."""

    def __init__(self, graph, backend="sim", rows=16, float_graph=None, verify=True, input_shape=None, **spec_kw):
        if backend not in ("sim", "qwen"):
            raise ValueError(f"backend must be 'sim' or 'qwen', got {backend!r}")
        self.graph, self.backend, self.rows, self.float_graph, self.verify = graph, backend, rows, float_graph, verify
        self._spec_kw = dict(spec_kw)
        self.lanes = spec_kw.pop("lanes", None)              # vectorized machine: neuron results per emitted token (docs/VECTORIZED.md)
        try:
            if self.lanes:
                from .machine.vec import compile_vec
                self.comp = compile_vec(graph, rows=rows, lanes=self.lanes, **spec_kw)
            else:
                self.comp = compile_graph(graph, rows=rows, **spec_kw)
        except _SchedDoesNotFit as e:
            raise _fit_error(graph, e) from None
        self.spec = self.comp.spec
        if self.lanes and max(len(c) for c in self.spec.cls_cols) > MAX_COLS:
            raise _fit_error(graph, f"a lane class needs {max(len(c) for c in self.spec.cls_cols)} physical columns but a mask head holds at most {MAX_COLS}")
        if not self.lanes and self.spec.ncols > MAX_COLS:
            raise _fit_error(graph, f"the program needs {self.spec.ncols} physical columns (weight columns, one or more per neuron per touched tile) "
                                    f"but a program-memory mask head holds at most {MAX_COLS}")
        self._machine = None
        self.input_shape = input_shape
        self.last = {}

    # ------------------------------------------------------------------ execution
    def _backend_machine(self):
        if self._machine is None:
            if self.backend == "sim":
                if self.lanes:
                    from .machine.vec import VecMachine
                    self._machine = VecMachine(self.spec)
                else:
                    from .machine.run import IRMachine
                    self._machine = IRMachine(self.spec)
            else:
                self._machine = _QwenMachine(self.spec)
        return self._machine

    def _execute(self, xq):
        """One encoded sample -> (output vector, prompt length, emitted tokens).  The host only builds the prompt and decodes."""
        prog = self.comp.program(xq); prompt = prog.prompt()
        m = self._backend_machine()
        if self.backend == "sim":
            # teacher forcing on the model's own predictions == autoregression for a deterministic argmax (see run.IRMachine)
            em = prog.expected(); pred = m.run(prompt, em)
            if pred != em + [self.spec.HALT]:
                raise MachineMismatch("circuit simulator disagrees with the token golden (executor bug)")
            emitted = pred[:-1]
        else:
            from .machine.run import generate
            emitted = generate(m.feed_fn(), prompt, self.spec.HALT)
        return self.comp.decode(emitted), len(prompt), len(emitted)

    def encode(self, x):
        return R.encode_input(self.graph, x)

    def run(self, x, check=None):
        """x: raw samples (N, *input_shape) (or one sample). Returns int output vectors (N, out_len) as decoded from the machine.
        check (default: self.verify) asserts machine output == reference.run for every sample."""
        x = np.asarray(x)
        if self.input_shape is not None and x.ndim == len(self.input_shape): x = x[None]     # a single sample
        xq = self.encode(x)
        ys, plen, nem = [], 0, 0
        for v in xq:
            y, plen, ne = self._execute(v); ys.append(y); nem += ne
        y = np.stack(ys)
        self.last = dict(prompt_tokens=plen, emitted_tokens_total=nem, samples=len(xq))
        if self.verify if check is None else check:
            ref = R.run(self.graph, xq)
            bad = np.nonzero((y != ref).any(1))[0]
            if len(bad): raise MachineMismatch(f"machine != reference.run on {len(bad)}/{len(y)} samples (first: {bad[0]}): {y[bad[0]]} vs {ref[bad[0]]}")
        return y

    def predict(self, x, check=None):
        y = self.run(x, check)
        return y.argmax(1)

    def check(self, x):
        """Assert machine == reference.run on x; returns True (raises MachineMismatch otherwise)."""
        self.run(x, check=True); return True

    def reference(self, x):
        return R.run(self.graph, self.encode(x))

    def accuracy(self, x, y, on="machine"):
        """Accuracy of 'machine' (runs the backend), 'int' (reference.run) or 'float' (the float IR)."""
        y = np.asarray(y)
        if on == "machine": return float((self.predict(x) == y).mean())
        if on == "int": return float((self.reference(x).argmax(1) == y).mean())
        if on == "float":
            if self.float_graph is None: raise ValueError("no float graph kept (compiled from an integer Graph)")
            return quant.float_accuracy(self.float_graph, x, y)
        raise ValueError(on)

    # ------------------------------------------------------------------ introspection
    def cost(self):
        """Static cost: IR tile-matvec model plus the machine program size."""
        sp = self.spec; c = self.graph.cost(sp.rows, sp.rows)
        zero = self.comp.program(np.zeros(len(self.comp.in_regs), dtype=np.int64))
        return dict(neurons=len(self.comp.neurons), ops=len(self.comp.groups if self.lanes else self.comp.ops), registers=sp.nreg, tiles=sp.tiles, columns=sp.ncols,
                    weight_cells=len(self.comp.W), prompt_tokens_empty_input=len(zero.prompt()), spec=str(sp),
                    ir=c, total_tile_matvecs=c["total_tile_matvecs"])

    def explain(self):
        g, sp = self.graph, self.spec; c = self.cost()
        lines = [f"qwentensor Program  backend={self.backend}  profile={g.profile} u{g.ab}s{g.wb} bias_rows={g.bias_rows}",
                 g.summary(), "",
                 (f"machine: {sp.tiles} tile(s) x {sp.rows} rows, {sp.lanes} lanes, columns/tile {sp.cols}, fan {sp.fan}, smax {sp.smax}" if self.lanes else
                  f"machine: {sp.tiles} tile(s) x {sp.rows} rows, columns/tile {sp.cols}, fan {sp.fan}, nops {sp.nops}, smax {sp.smax}"),
                 f"program: {c['neurons']} neurons = {c['ops']} ops, {c['registers']} registers, {c['columns']} columns, "
                 f"{c['weight_cells']} weight cells, prompt ~{c['prompt_tokens_empty_input']} tokens (+1 per nonzero input)",
                 (f"each op group ({self.lanes} neurons) = one generated token; the host loop is argmax-and-append, nothing else." if self.lanes else
                  "each op = one generated token (ACT of the result); the host loop is argmax-and-append, nothing else."),
                 f"pipeline: from_torch -> quantize{'+qat' if g.meta.get('qat_train_acc') is not None else ''} -> compile_graph -> prompt -> generate -> decode == reference.run"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Save the integer graph + compile options (the machine program is recompiled deterministically on load)."""
        d = self.graph.to_dict()
        with open(path, "w") as f:
            json.dump(dict(qwentensor_program=1, backend=self.backend, rows=self.rows, spec_kw=self._spec_kw, graph=d), f)

    @staticmethod
    def load(path, backend=None):
        with open(path) as f: d = json.load(f)
        if "qwentensor_program" not in d: raise ValueError(f"{path} is not a saved qwentensor Program")
        kw = dict(d["spec_kw"])
        if kw.pop("paged", False): return PagedProgram(Graph.from_dict(d["graph"]), backend or d["backend"], d["rows"], **kw)
        return Program(Graph.from_dict(d["graph"]), backend or d["backend"], d["rows"], **kw)


class PagedProgram(Program):
    """A compiled network larger than one machine: pages of neurons on one shared Spec (docs/STREAMING.md).  Same interface as Program.
    The model runs every page from its own tokens; the host carries the integer activations between pages."""

    def __init__(self, graph, backend="sim", rows=16, float_graph=None, verify=True, input_shape=None, max_cols=60, max_ops=60):
        if backend not in ("sim", "qwen"):
            raise ValueError(f"backend must be 'sim' or 'qwen', got {backend!r}")
        self.graph, self.backend, self.rows, self.float_graph, self.verify = graph, backend, rows, float_graph, verify
        self._spec_kw = dict(paged=True, max_cols=max_cols, max_ops=max_ops)
        try:
            self.paged = _pages.compile_paged(graph, rows=rows, max_cols=max_cols, max_ops=max_ops)
        except _SchedDoesNotFit as e:
            raise DoesNotFit(f"{e} (paged: max_cols={max_cols}, max_ops={max_ops}). See docs/STREAMING.md section 3.2 for the limits paging does not remove.") from None
        self.spec = self.paged.spec
        self._machine = None; self._exec = None
        self.input_shape = input_shape
        self.last = {}

    def _executor(self):
        if self._exec is None:
            if self.backend == "sim":
                self._exec = _pages.sim_executor(self.spec)
            else:
                self._machine = _QwenMachine(self.spec)
                self._exec = _pages.model_executor(self.spec, self._machine.feed_fn)
        return self._exec

    def _execute(self, xq):
        try:
            y, sizes = self.paged.run(xq, self._executor())
        except _pages.PageMismatch as e:
            raise MachineMismatch(str(e)) from None
        return y, sum(a for a, _ in sizes), sum(b for _, b in sizes)

    def run(self, x, check=None):
        y = super().run(x, check)
        self.last["pages"] = len(self.paged.pages)
        return y

    def cost(self):
        sp = self.spec; st = self.paged.stats()
        return dict(neurons=st["neurons"], ops=st["neurons"], registers=sp.nreg, tiles=sp.tiles, columns=sp.ncols, pages=st["pages"],
                    weight_cells=sum(st["page_cells"]), page_neurons=st["page_neurons"], page_weight_cells=st["page_cells"], spec=str(sp),
                    chip_layers=st["layers"], vocab=st["nvocab"])

    def explain(self):
        g, sp = self.graph, self.spec; c = self.cost()
        return "\n".join([f"qwentensor PagedProgram  backend={self.backend}  profile={g.profile} u{g.ab}s{g.wb} bias_rows={g.bias_rows}", g.summary(), "",
                          f"machine (shared by every page): {sp.tiles} tile(s) x {sp.rows} rows, columns/tile {sp.cols}, fan {sp.fan}, nops {sp.nops}, smax {sp.smax}, "
                          f"{c['chip_layers']} chip layers, vocab {c['vocab']}",
                          f"program: {c['neurons']} neurons in {c['pages']} pages {c['page_neurons']}, {c['weight_cells']} weight cells in total",
                          "each page is a fresh prompt (weights, program, inputs); the model emits every op's ACT token (argmax; append) and the host copies "
                          "the integer outputs into the next page's inputs (docs/STREAMING.md)."])

    def save(self, path):
        d = self.graph.to_dict()
        with open(path, "w") as f:
            json.dump(dict(qwentensor_program=1, backend=self.backend, rows=self.rows, spec_kw=self._spec_kw, graph=d), f)


class _QwenMachine:
    """Real Qwen3.5 through qwentpu (CUDA).  Builds and caches the merged model for this machine spec."""

    def __init__(self, spec):
        import torch
        src = os.environ.get("QWEN_DIR")
        if not src: raise RuntimeError("backend='qwen' needs QWEN_DIR (path to the Qwen3.5 checkpoint). Use backend='sim' on CPU.")
        if not torch.cuda.is_available(): raise RuntimeError("backend='qwen' needs a CUDA GPU (A100). Use backend='sim' on CPU.")
        from qwentpu.merge import build_merged, instantiate
        if hasattr(spec, "lanes"):
            from .machine.vec import install_vec as install
        else:
            from .machine.gpu import install
        install(spec)
        tc, sd, v0, lay, g = build_merged(src, mute_qwen=False)
        self.model, self.v0 = instantiate(tc, sd, "cuda"), v0

    def feed_fn(self):
        from qwentpu.driver import TpuSession
        return TpuSession(self.model, self.v0, fast=True, readout="code").feed


def compile(model, calib_data=None, input_shape=None, *, profile="v2.1", backend="sim", labels=None, qat=None, ab=5, wb=5,
            bias_rows=1, rows=16, input_posneg=False, input_kind="float", input_max=None, verify=True, paged=False, max_cols=60,
            max_ops=60, **spec_kw):
    """Compile a torch.nn.Module (or a float/integer ir.Graph) to a machine Program.
    calib_data: samples used for quantization calibration (N, *input_shape).  profile: "v2.1" (shift 0..15) or "v2.0" (no shift).
    qat: None | True | dict(epochs=..., lr=...) -> STE fine-tune (needs labels).
    lanes=N (extra keyword): vectorized machine, N neuron results per emitted token and no READ token (docs/VECTORIZED.md).
    paged=True: a network larger than one machine runs as pages of at most max_cols columns / max_ops ops on one shared machine
    (weight streaming through the prompt; docs/STREAMING.md); returns a PagedProgram with the same interface."""
    fg, g = _to_graph(model, calib_data, input_shape, profile, ab, wb, bias_rows, qat, labels, input_posneg, input_kind, input_max)
    if paged:
        if spec_kw: raise TypeError(f"paged=True takes max_cols / max_ops, not {sorted(spec_kw)}")
        return PagedProgram(g, backend=backend, rows=rows, float_graph=fg, verify=verify, input_shape=input_shape, max_cols=max_cols, max_ops=max_ops)
    return Program(g, backend=backend, rows=rows, float_graph=fg, verify=verify, input_shape=input_shape, **spec_kw)


def load(path, backend=None):
    return Program.load(path, backend)
