<p align="center"><img src="viz/images/architecture.png" alt="architecture: nn.Module to torch.fx to quantize to integer IR to prompt to Qwen3.5" width="100%"></p>

<p align="center"><img src="viz/images/prompt_layout.png" alt="prompt layout: weights, program and input tokens, then the tokens Qwen emits" width="100%"></p>

The diagram is the CPU-simulator build of the digits MLP (QAT, shifts 4 and 4, 25 ops, 496-token prompt). The A100 run used a different quantization (shifts 2 and 4) and a 434-token prompt, so the token counts differ (CELL tokens are one per nonzero weight).

<p align="center"><img src="viz/images/quantization.png" alt="float weights to 5-bit integers, accuracy float vs PTQ vs QAT" width="100%"></p>
