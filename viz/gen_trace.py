#!/usr/bin/env python3
"""Build data/trace.json + data/trace.js (window.TRACE, window.RESULTS) from a REAL run of the qwentensor pipeline.

  python3 study.py        # once: trains the example nets, writes data/study.json + data/digits_graph.json (frontend PTQ/QAT numbers)
  python3 gen_trace.py    # frontend graph -> sched.compile_graph -> Program.prompt() -> CPU circuit-IR machine -> trace

What runs here (nothing is hard-coded): the quantized digits MLP (data/digits_graph.json) is compiled by qwentensor.machine.sched,
its prompt is built by Program.prompt(), the CPU circuit-IR machine (qwentpu's exact float64 evaluator, 9 chip layers) predicts the
next token at every position, and we assert  (a) predictions == Program.expected() + HALT, (b) decoded outputs == reference.run,
for 3 held-out digits. The first digit is written to the trace.  `source` says "cpu-simulator" unless data/results.json
has machine_run="real-model" (set it only after ~/qwentensor_run/{digits,general}.log on lambda-final show a verified run).
Tokens per kind follow qwentensor/machine/spec.py: CELL (weights), PROG (PS_READ/PS_ACT/PM/PSH program-memory words), ACT, READ.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(HERE, "data")
sys.path[:0] = [ROOT, os.path.join(ROOT, "examples"), *[p for p in [os.environ.get("QWENTPU_SRC")] if p]]
import numpy as np, torch, torch.nn as nn, torch.fx
from common import digits
from qwentensor import reference as R
from qwentensor.ir import Graph
from qwentensor.machine.sched import compile_graph
from qwentensor.machine.run import IRMachine, generate

CODE = ["model = nn.Sequential(", "    nn.Linear(15, 15), nn.ReLU(),", "    nn.Linear(15, 10))", "train_float(model, Xtr, ytr)"]
RUN_CODE = ["g = frontend.from_torch(model, (15,))", "q = quant.qat(quant.quantize(g, Xtr, cfg), Xtr, ytr)", "prog = compile_graph(q).program(x)", "out = generate(qwen.feed, prog.prompt())"]

REAL_CHECK = dict(where="lambda-final:~/qwentensor_run/digits.log (and general.log)", model="real merged Qwen3.5-0.8B on an A100",
    result="digits MLP 15-15-10 (older QAT graph, shifts 2,4): 40/40 exact on 20 digits (code and lm_head readouts): emitted tokens == numpy token golden AND outputs == reference.run; general machine: 4 programs x 6/6 exact",
    note="that run is a different trained graph than the one in this trace (shifts 4,4), so this trace stays labelled cpu-simulator")

def js_only():
    T = json.load(open(os.path.join(DATA, "trace.json"))); res = json.load(open(os.path.join(DATA, "results.json"))); res["real_model_check"] = REAL_CHECK
    json.dump(res, open(os.path.join(DATA, "results.json"), "w"), indent=1)
    open(os.path.join(DATA, "trace.js"), "w").write("window.TRACE=%s;window.RESULTS=%s;" % (json.dumps(T), json.dumps(res))); print("rebuilt trace.js")

def main():
    if "--js-only" in sys.argv: return js_only()
    st = json.load(open(os.path.join(DATA, "study.json"))); D = st["digits"]; Dp = st["deep"]
    g = Graph.load(os.path.join(DATA, "digits_graph.json"))
    _, Xte, _, yte = digits("pool15"); xq_all = R.encode_input(g, Xte); ref_all = R.run(g, xq_all)
    int_acc = float((ref_all.argmax(1) == yte).mean())
    comp = compile_graph(g, rows=16); sp = comp.spec; m = IRMachine(sp)
    # ---- verify 3 digits on the CPU machine: emitted == Program.expected, outputs == reference.run
    idxs = [i for i in range(len(yte)) if ref_all[i].argmax() == yte[i]][:3]; trace = None
    for i in idxs:
        prog = comp.program(xq_all[i]); p = prog.prompt(); em = prog.expected()
        assert m.run(p, em) == em + [sp.HALT], "machine != Program.expected"
        if trace is None:   # the REAL host loop (run.generate: argmax; append) driven by the machine, for the traced digit
            state = []
            def feed(new): state.extend(new); return m.next_tokens(state, len(state) - len(new))
            assert generate(feed, p, sp.HALT) == em, "generate() != Program.expected"
        out = comp.decode(em); assert (out == ref_all[i]).all(), "decoded != reference.run"
        if trace is None: trace = (i, prog, p, em, out)
        print("digit", i, "label", yte[i], "pred", out.argmax(), "prompt", len(p), "emitted", len(em), "OK", flush=True)
    i, prog, p, em, out = trace; x = xq_all[i]
    # ---- sections
    ncell = sum(sp.is_cell(t) for t in p); nprog = sum(sp.prog_dec(t) is not None for t in p); nact = sum(sp.is_act(t) for t in p)
    assert ncell + nprog + nact == len(p)
    kinds = ["CELL" if sp.is_cell(t) else "PROG" if sp.prog_dec(t) is not None else "ACT" for t in p]
    # ---- program memory (real words)
    ops = [dict(j=j, name=o.name, dest=o.dest, col=o.cols[0], shift=o.shift) for j, o in sorted(prog.ops.items())]
    chain = [prog.entry] + [prog.ops[j].dest for j in prog.order]; nxt = list(prog.order) + [None]
    succ = [dict(reg=r, next_op=j) for r, j in zip(chain, nxt)]
    # ---- real accumulator sums of the first matvec layer (for the shift scene)
    n0 = g.matvecs()[0]; K, N = n0.W.shape
    y0 = (x.astype(np.int64) @ n0.W + n0.b).tolist(); v0 = [int(np.clip(v >> n0.shift, 0, g.amax)) for v in y0]
    # emitted ACT values of layer 1 must equal v0
    acts = [(sp.act_reg(t), sp.act_val(t)) for t in em if sp.is_act(t)]
    assert [v for _, v in acts[:N]] == v0, "layer-1 sums do not match emitted tokens"
    emitted = []
    for t in em:
        if sp.is_read(t): j = t - sp.READ0; emitted.append(dict(kind="READ", k=j, val=0, name=prog.ops[j].name, id=int(t)))
        else: emitted.append(dict(kind="ACT", k=int(sp.act_reg(t)), val=int(sp.act_val(t)), id=int(t)))
    emitted.append(dict(kind="HALT", k=0, val=0, id=0))
    # ---- quantization scene: first-layer 4x6 corner, float vs PTQ ints (study.py), effective scale by least squares
    wf = np.array(D["float_w1"]); wq = np.array(D["ptq_w1q"]); k = float((wf * wq).sum() / (wf * wf).sum())
    fxm = torch.fx.symbolic_trace(nn.Sequential(nn.Linear(15, 15), nn.ReLU(), nn.Linear(15, 10)))
    fx = [dict(name=("x" if n.op == "placeholder" else n.name), op=n.op, target=str(n.target), detail="") for n in fxm.graph.nodes]
    fx[1].update(detail="Linear 15->15"); fx[2].update(detail="ReLU"); fx[3].update(detail="Linear 15->10"); fx[0]["detail"] = "x[15]"; fx[4]["detail"] = "logits[10]"
    layers = [dict(name=n.name, shape=list(n.W.shape), shift=int(n.shift)) for n in g.matvecs()]
    ir = ["input   x[15]    uint5 (host encodes)", *[f"MATVEC  {n.name} {n.W.shape[0]}x{n.W.shape[1]}  shift {n.shift}  relu_clamp" for n in g.matvecs()], "argmax  (host, over the last vector)"]
    real = json.load(open(os.path.join(DATA, "results.json"))).get("machine_run") == "real-model" if os.path.exists(os.path.join(DATA, "results.json")) else False
    T = dict(placeholder=False, source="real-model" if real else "cpu-simulator", model="digits MLP 15-15-10 (examples/digits_mlp.py)",
        code=CODE, run_code=RUN_CODE, fx=fx, layers=layers, ir=ir,
        quant=dict(bits=g.wb, k=k, w1f=wf[:4, :6].round(2).tolist(), w1q=wq[:4, :6].tolist(), shift=[l["shift"] for l in layers]),
        spec=dict(rows=sp.rows, cols=list(sp.cols), nops=sp.nops, smax=sp.smax, fan=sp.fan, nvocab=int(sp.nvocab), nreg=int(sp.nreg), layers=int(m.g.n_layers()), heads=int(m.stats()["heads"])),
        prompt=dict(cell=ncell, prog=nprog, act=nact, total=len(p), kinds=kinds, cells=[int(w) for (_, _), w in sorted(prog.W.items())]),
        progmem=dict(ops=ops, succ=succ, entry=int(prog.entry), act0=int(sp.ACT0), nout=int(sp.nout), nops=int(sp.nops)),
        emitted=emitted, x=[int(v) for v in x], accum=dict(y=[int(v) for v in y0], v=v0, shift=int(n0.shift), amax=int(g.amax), bias_range=[int(g.wlo), int(g.whi)]),
        result=dict(pred=int(out.argmax()), label=int(yte[i]), ref=int(ref_all[i].argmax()), out=[int(v) for v in out]),
        verified=dict(digits=len(idxs), what="CPU machine next-token == Program.expected() (3 digits); run.generate host loop == Program.expected() (traced digit); decoded outputs == reference.run"),
        study=st)
    json.dump(T, open(os.path.join(DATA, "trace.json"), "w"))
    res = dict(model=T["model"], n=D["n_test"], float_acc=D["float"], int_acc=D["v21_qat"], ptq_acc=D["v21_ptq"], qat_bias4_acc=D["v21_b4_qat"], ptq_bias4_acc=D["v21_b4_ptq"],
               ptq_v20_acc=D["v20_ptq"], trace_graph_acc=int_acc, deep=dict(float_acc=Dp["float"], v20_ptq=Dp["v20_ptq"], v21_ptq=Dp["v21_ptq"]),
               real_model_check=REAL_CHECK, machine_run="real-model" if real else "cpu-simulator", placeholder=False,
               note="accuracies are the bit-exact integer reference (reference.run) on 540 held-out digits; machine_run says where the token trace ran")
    rp = os.path.join(DATA, "results.json")
    if os.path.exists(rp):
        old = json.load(open(rp))
        if old.get("machine_run") == "real-model": res["machine_run"] = "real-model"
    json.dump(res, open(rp, "w"), indent=1)
    open(os.path.join(DATA, "trace.js"), "w").write("window.TRACE=%s;window.RESULTS=%s;" % (json.dumps(T), json.dumps(res)))
    print("wrote data/trace.json, results.json, trace.js; source", T["source"], "sections", ncell, nprog, nact, "total", len(p), "emitted", len(em))

if __name__ == "__main__": main()
