"""qwentensor executor: a register machine with a prompt-written program memory, running inside Qwen3.5 via qwenTPU.

spec.py     machine configuration + token layout (what is compiled into the weights)
circuit.py  the circuit (qwentpu IR) of the machine
asm.py      low-level program (ops, registers, weights) -> prompt tokens; independent numpy token-level reference
sched.py    qwentensor.ir.Graph -> Program (register / column allocation, op order)
run.py      generation loop (host: argmax + append) and the CPU circuit-IR simulator driver
See docs/EXECUTOR.md.
"""
