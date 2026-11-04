#!/usr/bin/env python3
"""Rebuild prompt_layout.svg and quantization.svg (architecture.svg is hand-kept). Numbers come from ../data/trace.json and results.json."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(H, "../data/trace.json"))); R = json.load(open(os.path.join(H, "../data/results.json")))
F = 'font-family="Helvetica Neue,Helvetica,Arial,sans-serif"'; M = 'font-family="Menlo,Monaco,monospace"'
B, BL, O, OL, G, GL, P, PL = "#2563eb", "#dbe7ff", "#f08a24", "#fff0dd", "#16a34a", "#dcf5e4", "#7c3aed", "#efe6fb"
nb = lambda s: s.replace("  ", "&#160;&#160;")
def head(t, sub): return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" width="1600" height="900"><rect width="1600" height="900" fill="#fff"/>'
  f'<rect x="40" y="30" width="200" height="50" rx="12" fill="#2563eb"/><text x="140" y="64" text-anchor="middle" {F} font-weight="800" font-size="28" fill="#fff">qwentensor</text>'
  f'<text x="264" y="66" {F} font-weight="800" font-size="36" fill="#141a26">{t}</text><text x="1560" y="64" text-anchor="end" {F} font-size="24" fill="#6b7488">{sub}</text>'
  '<line x1="40" y1="96" x2="1560" y2="96" stroke="#dfe4ec" stroke-width="2"/><defs><marker id="a" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#6b7488"/></marker></defs>')
def T(x, y, t, sz=24, c="#6b7488", w=600, a="middle", m=False): return f'<text x="{x}" y="{y}" text-anchor="{a}" {M if m else F} font-weight="{w}" font-size="{sz}" fill="{c}">{nb(t)}</text>'
def rect(x, y, w, h, f, c, sw=3, r=16): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{f}" stroke="{c}" stroke-width="{sw}"/>'
def arr(x1, y1, x2, y2, c="#6b7488"): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="4" marker-end="url(#a)"/>'
# ---- prompt layout (proportional to the real token counts)
p = D["prompt"]; tot = p["total"]; s = head("PROMPT LAYOUT", "digits MLP, real Program.prompt()")
s += rect(60, 170, 1480, 190, "#fbfcfe", "#dfe4ec", 2, 20) + T(90, 210, f"PROMPT  {tot} tokens, in this order", 24, "#6b7488", 800, "start")
x = 80; W = 1440
for lab, n, c, f, sub in [("WEIGHTS (CELL)", p["cell"], O, OL, "one token per nonzero weight"), ("PROGRAM (PROG)", p["prog"], P, PL, "succ, dest, mask, shift"), ("INPUT (ACT)", p["act"], G, GL, "register init")]:
    w = {'cell': 900, 'prog': 300, 'act': 240}[lab.split('(')[1][:-1].lower()]; s += rect(x, 230, w - 10, 110, f, c) + T(x + w / 2 - 5, 276, lab, 26, c, 800) + T(x + w / 2 - 5, 308, f"{n} tokens", 22, "#141a26", 700) + T(x + w / 2 - 5, 332, sub, 17); x += w
s += rect(60, 400, 1480, 210, "#fbfcfe", "#dfe4ec", 2, 20) + T(90, 442, f"EMITTED BY QWEN  {len(D['emitted'])} tokens, one per step", 24, "#6b7488", 800, "start")
E = D["emitted"]; items = [(f"READ op{e['k']}" if e["kind"] == "READ" else f"ACT r{e['k']}={e['val']}" if e["kind"] == "ACT" else "HALT", e["kind"]) for e in E[:4]] + [("...", "d")] + [(f"ACT r{E[-2]['k']}={E[-2]['val']}", "ACT"), ("HALT", "HALT")]
col = {"READ": ("#64748b", "#e6ebf3"), "ACT": (B, BL), "HALT": (P, PL), "d": ("#fff", "#fff")}; x = 90
for lab, k in items:
    c, f = col[k]; w = max(60, len(lab) * 17 + 36)
    if k != "d": s += rect(x, 490, w, 64, f, c, 3, 32)
    s += T(x + w / 2, 531, lab, 24, c if k != "d" else "#6b7488", 800, m=True); x += w + 16
s += T(90, 590, "next token = succ[slot(current token)] (+ the computed value bits after a READ).  Token ids: qwentensor/machine/spec.py", 20, "#6b7488", 600, "start")
s += T(80, 680, "READ opj: the model starts op j.   ACT r=v: it writes value v to register r.   HALT: the successor of the last op.", 22, "#6b7488", 600, "start")
s += T(80, 730, "The prompt is the only host input. After it the host does: argmax; append.", 22, "#141a26", 800, "start") + "</svg>"
open(os.path.join(H, "prompt_layout.svg"), "w").write(s)
# ---- quantization
q = D["quant"]; wf = q["w1f"][0][:3]; wq = q["w1q"][0][:3]
s = head("QUANTIZATION", "floats to 5-bit integers")
s += rect(60, 200, 520, 200, BL, B) + T(320, 280, "w (float)", 34, B, 800, m=True) + T(320, 340, "  ".join(f"{v:.2f}" for v in wf) + " ...", 26, "#6b7488", 600, m=True)
s += rect(1020, 200, 520, 200, OL, O) + T(1280, 280, "q (int5)", 34, O, 800, m=True) + T(1280, 340, "  ".join(str(v) for v in wq) + " ...", 26, "#6b7488", 600, m=True)
s += arr(600, 300, 1000, 300, B) + T(800, 268, "q = clamp(round(w * k))", 28, B, 800, m=True) + T(800, 345, f"k = {q['k']:.2f} for this layer", 24) + T(800, 378, "k folds input scale, 2^shift, output scale", 20)
s += '<line x1="140" y1="560" x2="1460" y2="560" stroke="#cbd1dc" stroke-width="4"/>'
for v in range(-16, 16):
    xx = 140 + (v + 16) * 1320 / 31; h = 18 if v % 5 == 0 else 9
    s += f'<line x1="{xx}" y1="{560-h}" x2="{xx}" y2="{560+h}" stroke="#6b7488" stroke-width="2"/>'
    if v % 5 == 0: s += T(xx, 610, str(v), 22, "#6b7488", 700, m=True)
s += T(800, 500, "32 levels: every weight lands on one of these ticks", 26, "#141a26", 800)
s += rect(60, 660, 1480, 190, "#fbfcfe", "#dfe4ec", 2, 20) + T(90, 702, f"digits MLP 15-15-10, {R['n']} held-out digits (viz/data/results.json). Integer = bit-exact reference.run", 22, "#6b7488", 800, "start")
for i, (l, v, c) in enumerate([("float (torch)", R["float_acc"], "#cbd1dc"), ("integer, PTQ", R["ptq_acc"], "#8fb2f5"), ("integer, after QAT", R["int_acc"], B)]):
    y = 722 + i * 40; s += T(100, y + 24, l, 24, "#141a26", 700, "start") + rect(460, y, 800, 30, "#f5f7fb", "#dfe4ec", 1.5, 10) + rect(460, y, 800 * v, 30, c, c, 1, 10) + T(1290, y + 25, f"{v*100:.1f}%", 26, "#141a26", 800, "start", True)
open(os.path.join(H, "quantization.svg"), "w").write(s + "</svg>")
