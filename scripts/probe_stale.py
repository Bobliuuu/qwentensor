"""Probe: does the circuit bake act*weight at ACT time (so a later CELL overwrite cannot change an already-ACT'd row)?
Prompt = normal prompt with one extra CELL(row 0, col 0, new weight) inserted AFTER the input ACT tokens.  Also probes the
other half of in-model streaming: re-issuing ACT(r; 0) for a register overwrites its accumulator row (erase)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.environ.get("QWENTPU_SRC", os.path.expanduser("~/qwentpu_src/qwentpu"))]
import numpy as np
from qwentensor.ir import Graph, Node
from qwentensor.machine.run import IRMachine
from qwentensor.machine.sched import compile_graph
from qwentensor.machine.asm import reference_emit

W = np.array([[3], [1], [1]]); b = np.array([0])
g = Graph([Node("x", "input", [], 3), Node("fc", "matvec", ["x"], 1, W=W, b=b, shift=0)], "fc", ab=3, wb=3, profile="v2.1")
comp = compile_graph(g, rows=8); sp = comp.spec; m = IRMachine(sp)
x = np.array([2, 1, 1])                                  # y = 3*2 + 1 + 1 = 8 -> clamp 7
prog = comp.program(x); p = prog.prompt()
def circuit_value(prompt):
    """value the circuit emits for READ(0) when the prompt is followed by the READ token."""
    nxt = m.next_tokens(list(prompt) + [sp.read(0)], len(prompt))[0]
    return sp.act_val(nxt) if sp.is_act(nxt) else ("tok", nxt)
print("plain prompt, circuit READ(0) value:", circuit_value(p), " expected:", min(7, int(x @ W[:, 0])))
# overwrite weight of row 0 (the register of x0) to -3 AFTER all ACT tokens (insert before the last token = ACT(entry))
newcell = sp.cell(0, 0, -3)
p2 = p[:-1] + [newcell] + p[-1:]
print("weight overwritten after the ACT:", circuit_value(p2), " (static-weight golden would be", max(0, min(7, int(np.array([-3, 1, 1]) @ x))), ", baked-at-ACT would be 7)")
# overwrite BEFORE the ACT tokens (the normal case: CELL first)
first_act = next(i for i, t in enumerate(p) if sp.is_act(t))
p3 = p[:first_act] + [newcell] + p[first_act:]
print("weight overwritten before the ACT:", circuit_value(p3), " (golden with new weight:", max(0, min(7, int(np.array([-3, 1, 1]) @ x))), ")")
# erase by re-ACT: re-issue ACT(r0; 0) after the original ACT(r0; 2) (same register) -> accumulator row 0 should be rewritten with 0
r0 = comp.in_regs[0]
p4 = p[:-1] + [sp.act(r0, 0)] + p[-1:]
print("re-ACT(r0;0) after ACT(r0;2):", circuit_value(p4), " (erased row -> y = 0+1+1 = 2; not erased -> 7)")
