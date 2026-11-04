"""Drivers: the host loop (argmax; append) and the CPU circuit-IR simulator."""
import numpy as np

from qwentpu.compiler import evaluate as ir

from .circuit import build


def generate(feed, prompt, halt=0, max_new=100000):
    """THE host loop: feed(tokens) -> the model's argmax next token at each fed position.  Feed the prompt, then append the
    model's own token until it emits HALT.  Knows nothing about matrices, ops, registers or columns."""
    nxt = feed(list(prompt))[-1]
    out = []
    while nxt != halt and len(out) < max_new:
        out.append(nxt)
        nxt = feed([nxt])[-1]
    return out


class IRMachine:
    """The machine circuit evaluated by the exact float64 IR evaluator (every bf16 / silu / state exactness condition asserted).
    Teacher forcing on the model's own emitted tokens is identical to autoregression for a deterministic argmax."""

    def __init__(self, sp):
        self.sp = sp; self.g, self.d = build(sp)

    def next_tokens(self, toks, start=0):
        m = ir.evaluate(self.g, toks)
        cd = self.d["code"]
        return [sum((1 << i) * int(ir.ev(cd[i], m[t:t + 1])[0] > 0) for i in range(len(cd))) for t in range(start, len(toks))]

    def run(self, prompt, emitted):
        """Predictions at the last prompt position and at every emitted position."""
        return self.next_tokens(list(prompt) + list(emitted), len(prompt) - 1)

    def stats(self): return {**self.g.stats(), "layers": self.g.n_layers()}
