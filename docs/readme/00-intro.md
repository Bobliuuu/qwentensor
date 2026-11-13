Write a small neural network in PyTorch, quantize it to 5-bit integers, and run it as a program inside Qwen3.5 using the [qwenTPU](https://github.com/Bobliuuu/qwentpu) machinery.
The program, including the schedule, is written in the prompt; the host only runs `argmax; append`.

```python
import qwentensor as qt
net = qt.compile(model, calib_data, input_shape=(64,), backend="qwen")   # nn.Module in
print(net.predict(x))                                                    # class indices; the underlying outputs are bit-exact vs the numpy golden
```

✨ **At a glance**
* 🐍 any small `nn.Module` (Linear / ReLU / Conv / residual Add / pooling)
* 🔢 5-bit integer weights, per-layer shift requantize
* 📝 the schedule, successors and masks are **tokens in the prompt**: swap the prompt, swap the program
* ✅ bit-exact vs a numpy golden on a real Qwen3.5-0.8B (A100): **40/40** digits MLP, **24/24** across four different programs
* 🤖 inside one run the host does only `argmax; append`
* 📄 networks larger than one machine run in **pages** (`paged=True`): a 142-neuron deep MLP and a 426-neuron CNN ran exact on the real model, the host carrying activations between pages

<p align="center"><a href="docs/qwentensor_demo.mp4"><img src="docs/anim/hero.gif" alt="a PyTorch network compiled to a prompt that Qwen3.5 executes" width="100%"></a><br>
🎬 <a href="docs/qwentensor_demo.mp4"><b>Watch the 1:42 demo video</b></a> (hero, program memory, shift requantize)</p>

**📍 Status.** The frontend, quantizer and bit-exact numpy golden are done. The executor reproduces the golden bit-for-bit on a **CPU simulator of the circuit**
(qwenTPU's float64 circuit-IR evaluator). On the real model, programs from small MLPs to a 778-neuron CNN have been run exact (below). First, the quantized digits MLP 15-15-10 (u5s5, shifts 2 and 4) on
Qwen3.5-0.8B (A100, bf16): 20 held-out digits x 2 readouts (code bits and lm_head) = 40/40 exact, emitted tokens equal to the numpy token golden and outputs equal
to `reference.run` (prompt 434 tokens, 50 emitted, 9 chip layers, hidden 3072, vocab 17616; 95% argmax accuracy on those 20 digits). The host did only `argmax; append`.
Log: [results/a100_digits.log](results/a100_digits.log).
Second, **one compiled 14-layer model** (Spec rows=16, cols=(15,12,8), fan 2, smax 7, 40 ops, hidden 3712, vocab 26616, same A100 / bf16) ran four different programs,
6 inputs each, 24/24 exact (emitted tokens equal to the numpy token golden and outputs equal to `reference.run`); only the prompt differed:

| program | exact | prompt tokens | emitted |
|---|---|---|---|
| digits MLP 15-15-10 | 6/6 | 459 | 50 |
| random MLP 20-8-4 (K=20 spans 2 tiles) | 6/6 | 282 | 24 |
| random 4-layer MLP 8-5-5-5-4 | 6/6 | 259 | 38 |
| residual block (concat, identity rows) | 6/6 | 222 | 28 |

Log: [results/a100_general.log](results/a100_general.log).
Third, a **residual MLP 15-10-[10+res]-10** (the `examples/run_residual.py` network: concat plus identity rows, 40 ops over 4 tiles, 16 chip layers, hidden 4608, vocab 37184) on the same
0.8B: 20 digits x 2 readouts = 40/40 exact, prompt 685 tokens, 80 emitted, 90% argmax accuracy on those 20 digits
([results/a100_residual_q08.log](results/a100_residual_q08.log)). The frontend's 64-32-[32+res]-10 residual MLP, the deep MLP and the CNN do not fit one machine; they run in pages and are in the sections below.

**Bigger models.** The same two programs ran bit-exact on Qwen3.5-2B and 4B (A100, bf16); only the hidden size changes with the model:

| model | digits MLP (40 runs) | general, 4 programs (24 runs) | hidden (digits / general) | logs |
|---|---|---|---|---|
| 0.8B | 40/40 | 24/24 | 3072 / 3712 | [digits](results/a100_digits.log), [general](results/a100_general.log) |
| 2B | 40/40 | 24/24 | 4096 / 4864 | [digits](results/a100_digits_q2.log), [general](results/a100_general_q2.log) |
| 4B | 40/40 | 24/24 | 4608 / 5376 | [digits](results/a100_digits_q4.log), [general](results/a100_general_q4.log) |

**Throughput** (0.8B, A100, digits MLP, 20 inputs, host loop = `argmax; append`, one cache, 434-token prompt, 50 emitted tokens; all 20 exact in both modes,
[log](results/a100_throughput_q08.log)). The CUDA-graph chip path is qwenTPU's `GraphedChip`:

| chip path | prefill | emitted-token loop | end to end |
|---|---|---|---|
| eager | 24 ms | 657 ms = 76 tok/s (13.1 ms/token) | 681 ms (1.47 inferences/s) |
| CUDA graph | 102 ms | 163 ms = 307 tok/s (3.25 ms/token) | 265 ms (3.78 inferences/s) |

The graph path is 4.0x faster per emitted token and 2.6x faster per inference; its prefill is slower because it also captures the prompt shapes.
See [docs/EXECUTOR.md](docs/EXECUTOR.md) section 7.

Fourth, **networks larger than one machine run in pages** (weight streaming through the prompt, `qt.compile(..., paged=True)`, [docs/STREAMING.md](docs/STREAMING.md)). A page is
a normal machine program for a group of neurons; all pages share one machine; every page is a fresh prompt that the model runs from its own tokens (`argmax; append`), and the
host copies the integer outputs of one page into the next page's input tokens. On the real Qwen3.5-0.8B (A100, bf16), 10 held-out digits x 2 readouts per network, every
page's emitted tokens equal to the numpy token golden and every final output equal to `reference.run`, **60/60 exact, 1060 page runs, 0 mismatches**:

| network | neurons | pages | machine | prompt tokens per page | emitted (10 digits) | exact |
|---|---|---|---|---|---|---|
| residual MLP 44-20-[20+res]-20-10 | 70 | 7 | 44 columns, 15 chip layers, hidden 4352, vocab 28992 | 161-574 | 1400 | 20/20 |
| deep MLP 44-44-44-44-10 | 142 (> 128 op slots) | 10 | 45 columns, 15 chip layers, hidden 4352, vocab 29279 | 529-749 | 2840 | 20/20 |
| strided CNN 1x8x8, conv 4, conv 8, conv 8, fc 10 | 426 | 36 | 42 columns, 15 chip layers, hidden 4096, vocab 31872 | 157-452 | 8520 | 20/20 |

Log: [results/a100_stream.log](results/a100_stream.log). Between pages the activations live in the host, so the page sequence is host-driven; inside a page nothing is.

Fifth, **wide neurons** ([docs/WIDE.md](docs/WIDE.md)): the three original examples whose neurons read 64, 96 and 512 registers, and a 784-wide first layer, run in pages with
tile groups on the same real Qwen3.5-0.8B (A100, bf16). Every page's emitted tokens equal the numpy token golden and every final output equals `reference.run`:

| network | neurons | pages | machine | prompt tokens per page | digits x readouts | exact | page runs |
|---|---|---|---|---|---|---|---|
| residual MLP 64-32-[32+res]-10 | 106 | 14 | 43 columns, 19 chip layers, hidden 4352, vocab 28956, 2 tile groups | 120-508 | 10 x 2 | 20/20 | 280 |
| deep MLP 64-96-64-32-10 | 202 | 37 | 41 columns, 21 chip layers, hidden 4352, vocab 28402, 3 tile groups | 236-610 | 10 x 2 | 20/20 | 740 |
| CNN conv4-conv8-AvgPool-Linear(128->10) (512-wide Linear) | 778 | 110 | 58 columns, 33 chip layers, hidden 8704, 11 tile groups | 744-1382 | 6 x 2 | 12/12 | 1320 |
| seeded random 784-8-4 network (no 784-input dataset offline; checked only against `reference.run`) | 12 | 9 | 56 columns, 40 chip layers, hidden 9984, vocab 75588, 18 tile groups | 1109-1690 | 3 x 2 | 6/6 | 54 |

Logs: [residual](results/a100_wide_residual.log), [deep](results/a100_wide_deep.log), [CNN](results/a100_wide_cnn.log), [784](results/a100_wide784.log). The digit
runs use the first 10 (CNN: 6) held-out digits; the 784-wide network has random weights and random inputs, so it shows exactness, not accuracy.

Sixth, a **vectorized machine** ([docs/VECTORIZED.md](docs/VECTORIZED.md), `qt.compile(..., lanes=N)`, `scripts/gpu_vec.py`): the READ token is fused into the register write and one emitted token carries
`lanes` neuron results, so the digits MLP emits 25 tokens (one lane) or 13 (two lanes) instead of 50. Same 0.8B, A100, digits MLP, 20 digits x 2 readouts = **40/40 exact** for each of one and two lanes
([log](results/a100_vec_exact.log)); one compiled one-lane machine (16 chip layers, hidden 4608) ran four different programs, **24/24 exact** ([log](results/a100_vec_general.log)). The scalar path is unchanged and stays the default.
Throughput, same protocol as above (20 inputs, all exact in both modes, [log](results/a100_vec_bench.log)):

| machine | chip layers | emitted tokens | eager loop | eager end to end | CUDA-graph loop | CUDA-graph end to end |
|---|---|---|---|---|---|---|
| scalar (above) | 9 | 50 | 656 ms (76 tok/s) | 680 ms | 163 ms (308 tok/s) | 266 ms |
| vectorized, 1 lane | 9 | 25 | 331 ms (76 tok/s) | 355 ms | 101 ms (249 tok/s) | 205 ms |
| vectorized, 2 lanes | 17 | 13 | 322 ms (40 tok/s) | 367 ms | 146 ms (89 tok/s) | 340 ms |

One lane halves the tokens at the same circuit size: 2.0x faster eager, 1.6x faster in the graph loop and 1.3x per inference (the graph path's 100 ms prefill is not reduced). Two lanes cut the tokens to 13 but need two columns per neuron (56 columns, 121 heads, 17 layers, vocab 71840), so
a token costs 2x more; it ties one lane eagerly (322 ms) and is slower with CUDA graphs. One lane is the faster configuration for this network.
