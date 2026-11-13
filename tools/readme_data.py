#!/usr/bin/env python3
"""Derive results/derived/readme_data.json (used by README.spec.yml) by parsing the committed A100 logs in results/a100_*.log.
Nothing is typed by hand: every count, token number and timing below is a regex match on those logs."""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: open(os.path.join(ROOT, "results", f)).read()


def exact_lines(text):
    """'[tag] name: A/B exact ...; prompt P tok, emitted E' -> rows"""
    rows = []
    for m in re.finditer(r"\[(?P<tag>[\w:]+)\] (?P<name>[^\n]*?): (?P<a>\d+)/(?P<b>\d+) exact[^\n]*?prompt (?P<p>\d+) tok, emitted (?P<e>\d+)", text):
        rows.append(dict(tag=m["tag"], name=m["name"], exact=int(m["a"]), total=int(m["b"]), prompt=int(m["p"]), emitted=int(m["e"])))
    return rows


def main():
    models = []
    for label, d, g in [("Qwen3.5-0.8B", "a100_digits.log", "a100_general.log"), ("Qwen3.5-2B", "a100_digits_q2.log", "a100_general_q2.log"), ("Qwen3.5-4B", "a100_digits_q4.log", "a100_general_q4.log")]:
        dt, gt = R(d), R(g)
        dm = re.search(r"digits MLP[^\n]*?: (\d+)/(\d+) exact[^\n]*", dt)
        hid = re.search(r"hidden (\d+)", dt).group(1)
        grows = exact_lines(gt)
        models.append(dict(key={"Qwen3.5-0.8B": "m08", "Qwen3.5-2B": "m2", "Qwen3.5-4B": "m4"}[label], model=label, digits_exact=int(dm[1]), digits_total=int(dm[2]), general_exact=sum(r["exact"] for r in grows), general_total=sum(r["total"] for r in grows),
                           hidden=int(hid), bad=int(re.search(r"TOTAL BAD (\d+)", dt)[1]) + int(re.search(r"TOTAL BAD (\d+)", gt)[1])))
    programs = [dict(program=r["name"], exact=r["exact"], total=r["total"], prompt=r["prompt"], emitted=r["emitted"]) for r in exact_lines(R("a100_general.log"))]

    paged = []
    for f in ["a100_stream.log", "a100_wide_residual.log", "a100_wide_deep.log", "a100_wide_cnn.log", "a100_wide784.log"]:
        t = R(f)
        for m in re.finditer(r"\[(?P<tag>[\w:]+)\] readout (?P<ro>code|lm_head): (?P<a>\d+)/(?P<b>\d+) digits exact end to end[^\n]*?; (?P<pages>\d+) pages run, (?P<mm>\d+) page mismatches; prompt (?P<p0>\d+)\.\.(?P<p1>\d+) tok, (?P<emit>\d+) tokens emitted", t):
            tag = m["tag"]
            paged.append(dict(network=tag.replace(":", " "), readout=m["ro"], source=f, digits_exact=int(m["a"]), digits_total=int(m["b"]), pages_run=int(m["pages"]), page_mismatches=int(m["mm"]),
                              prompt_min=int(m["p0"]), prompt_max=int(m["p1"]), emitted=int(m["emit"])))
    agg = {}
    for p in paged:
        a = agg.setdefault(p["network"], dict(network=p["network"], digit_runs=0, exact=0, pages_run=0, emitted=0, mismatches=0))
        a["digit_runs"] += p["digits_total"]; a["exact"] += p["digits_exact"]; a["pages_run"] += p["pages_run"]; a["emitted"] += p["emitted"]; a["mismatches"] += p["page_mismatches"]
    paged_networks = list(agg.values())
    t = R("a100_throughput_q08.log")
    thr = []
    for m in re.finditer(r"(eager|CUDA-graph) chip path: exact (\d+)/(\d+); prompt (\d+) tok, emitted (\d+) tok; prefill ([\d.]+) ms; emitted-token loop ([\d.]+) ms = ([\d.]+) tok/s \(([\d.]+) ms/token\); end-to-end ([\d.]+) ms per inference \(([\d.]+) inf/s\)", t):
        thr.append(dict(path=m[1], exact=int(m[2]), total=int(m[3]), prefill_ms=float(m[6]), loop_ms=float(m[7]), tok_s=float(m[8]), ms_per_token=float(m[9]), e2e_ms=float(m[10]), inf_s=float(m[11])))
    vec = []
    for m in re.finditer(r"\[(scalar|vec1|vec2):bench\] (eager|CUDA-graph) chip path: exact (\d+)/(\d+); layers (\d+); prompt (\d+) tok, emitted (\d+) tok; prefill ([\d.]+) ms; emitted-token loop ([\d.]+) ms = ([\d.]+) tok/s \(([\d.]+) ms/token\); end-to-end ([\d.]+) ms", R("a100_vec_bench.log")):
        vec.append(dict(machine={"scalar": "scalar", "vec1": "vectorized, 1 lane", "vec2": "vectorized, 2 lanes"}[m[1]], path=m[2], exact=int(m[3]), emitted=int(m[7]), layers=int(m[5]), tok_s=float(m[10]), e2e_ms=float(m[12]), series=f"{m[1]} {m[2]}"))
    models_long = [dict(model=m["model"], test=t, exact=m[k]) for m in models for t, k in (("digits MLP", "digits_exact"), ("general (4 programs)", "general_exact"))]
    out = dict(models=models, models_long=models_long, programs=programs, paged=paged, paged_networks=paged_networks, throughput=thr, vec=vec,
               totals=dict(digits_total_runs=sum(m["digits_total"] for m in models), digits_runs=sum(m["digits_total"] for m in models), digits_exact=sum(m["digits_exact"] for m in models), general_runs=sum(m["general_total"] for m in models),
                           general_exact=sum(m["general_exact"] for m in models), page_runs=sum(p["pages_run"] for p in paged), page_mismatches=sum(p["page_mismatches"] for p in paged),
                           paged_digits_exact=sum(p["digits_exact"] for p in paged), paged_digits_total=sum(p["digits_total"] for p in paged)))
    dst = os.path.join(ROOT, "results/derived/readme_data.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst, out["totals"])


if __name__ == "__main__":
    sys.exit(main())
