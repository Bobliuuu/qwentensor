"""The ORIGINAL examples (64-96-64-32-10 deep MLP, 64-32-[32+res]-10 residual MLP, 2-conv + AvgPool + 512-wide Linear CNN), whose neurons read
up to 512 registers, run in pages with tile groups (docs/WIDE.md).
usage: python examples/run_wide.py [deep|residual|cnn] [n_circuit]      (default: deep 1)

Loads the quantized graph written by scripts/make_wide_data.py and checks the machine against reference.run on
  * all 540 held-out samples with the numpy token golden of every page (fast), and
  * the first n_circuit samples through the CPU circuit-IR simulator (every page, every exactness condition asserted)."""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), *[p for p in [os.environ.get("QWENTPU_SRC")] if p]]
import numpy as np
from qwentensor import reference as R
from qwentensor.ir import Graph
from qwentensor.machine.pages import compile_paged, golden_executor, sim_executor

which = sys.argv[1] if len(sys.argv) > 1 else "deep"
n_circuit = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DATA = os.path.join(HERE, "..", "qwentensor", "machine", "data")
g = Graph.load(os.path.join(DATA, f"wide_{which}.json")); z = np.load(os.path.join(DATA, f"wide_samples_{which}.npz"))
pc = compile_paged(g, rows=16, max_cols=45, max_ops=60); st = pc.stats()
print(f"{which}: {st['neurons']} neurons in {st['pages']} pages; machine {pc.spec}; {st['layers']} chip layers, vocab {st['nvocab']}")
xq, ref = z["xq"], z["ref"]
print(f"integer accuracy (reference.run), {len(xq)} held-out samples: {(ref.argmax(1) == z['y']).mean():.4f}")
ex = golden_executor(); t = time.time()
out = np.array([pc.run(x, ex)[0] for x in xq])
print(f"paged machine (token golden) == reference.run on {len(xq)} samples: {bool((out == ref).all())} ({time.time()-t:.0f}s)")
sx = sim_executor(pc.spec); t = time.time()
for i in range(n_circuit):
    y, sizes = pc.run(xq[i], sx)
    assert (y == ref[i]).all()
print(f"paged machine (circuit simulator, every page checked against the token golden) == reference.run on {n_circuit} samples: True ({time.time()-t:.0f}s)")
