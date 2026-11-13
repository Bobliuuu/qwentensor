| | what runs | who decides the computation |
|---|---|---|
| qwenTPU v1 | one fixed matvec | the host drives the array one operation at a time |
| qwenTPU v2 | a hardcoded chain of matvec + requantize stages | the chain is compiled in; the program only varies widths and depth |
| **qwentensor** | any quantized network of the supported ops, written as a PyTorch `nn.Module` | the prompt: each op's successor, destination register, column mask and shift are program-memory tokens |

```
nn.Module --torch.fx--> float IR --quantize (PTQ, optional STE QAT)--> integer IR --+--> reference.run   (numpy golden, bit-exact)
                                                                                    +--> machine/        (general executor: prompt-written program, tiling, partial sums)
```

<p align="center"><img src="docs/anim/pipeline.gif" alt="the qwentensor pipeline from nn.Module to emitted tokens" width="100%"></p>

**Program memory.** The prompt writes a successor table into DeltaNet state, and the next token is looked up from it, so the program is prompt data:

<p align="center"><img src="docs/anim/program_memory.gif" alt="PROG tokens fill head state, then next token = succ[slot(current)]" width="100%"></p>

**Why a shift requantize and bias rows matter.** Without them the integer network collapses; with them it recovers:

<p align="center"><img src="docs/anim/shift_and_bias.gif" alt="clamped sums with and without a shift, bias rows 1 versus 4" width="100%"></p>


| piece | path | status |
|---|---|---|
| IR contract (ops, ranges, lowering table, supported/unsupported ops) | `docs/IR.md`, `qwentensor/ir.py` | done |
| torch.fx frontend | `qwentensor/frontend.py` | done |
| post-training quantizer + STE fine-tuning | `qwentensor/quant.py` | done |
| bit-exact numpy interpreter (the golden) | `qwentensor/reference.py` | done |
| executor: program memory, tiling, shift requant, bias rows | `qwentensor/machine/`, `docs/EXECUTOR.md` | bit-exact on the CPU simulator and on the real Qwen3.5-0.8B (A100): the digits MLP, small programs (K-tiled MLP, 4-layer MLP, residual blocks), and, in pages, the deep MLP, residual MLP and CNN including their 64-, 96- and 512-wide layers |
| user API `qt.compile(...)` | `qwentensor/api.py` | done (`backend="sim"`); `backend="qwen"` needs a GPU (digits, general, residual, stream, wide and wide784 phases ran on the real model via `scripts/gpu_run.py`; the vectorized machine via `scripts/gpu_vec.py`) |
| examples, tests | `examples/`, `tests/` | done (83 CPU tests) |

Design write-up: [docs/DESIGN.md](docs/DESIGN.md). Project page: `docs/site/`.
