Integer accuracy = bit-exact `reference.run` of the quantized IR (what the machine must reproduce). "PTQ" = post-training quantization with 512 calibration
samples taken from the training split; "+QAT" = STE fine-tuning of the integer weights, model selection on TRAIN accuracy only.

| network | float (torch) | v2.0 (no shift), bias 1 row: PTQ / +QAT | v2.1 (shift), bias 1 row: PTQ / +QAT | v2.1, bias 4 rows: PTQ / +QAT | tile matvecs on 16x16 |
|---|---|---|---|---|---|
| digits MLP 15-15-10 (qwenTPU's) | 90.4% | 43.7% / 82.6% (input 0..12); **87.6%** with qwenTPU-style QAT from scratch (250 ep x 4 seeds) | 76.3% / 89.4% | 85.2% / 89.8% | 4 |
| deep MLP 64-96-64-32-10 | 97.8% | 24.1% / 78.9% | 97.6% / 97.6% | 97.8% / 97.8% | 83 |
| CNN conv4-conv8-avgpool-fc (8x8) | 97.4% | 14.3% / 14.3% (QAT fails) | 87.6% / 97.2% | 96.5% / 97.0% | 299 |
| residual MLP 64-32-[32+res]-10 | 97.4% | 49.4% / 83.7% | 96.1% / 96.1% | 96.9% / 96.9% | 33 |

Executor status per row. The digits MLP was compiled with its real quantized weights and run through the machine (tests, shifts 2 and 4: output equal to `reference.run` in the circuit simulator, and exact on the real Qwen3.5-0.8B on an A100, in both the 9-layer and the 14-layer machine). The deep MLP, the CNN and the frontend's residual MLP exceed one machine and run in pages with tile groups for their wide layers (Status, "wide neurons"): on the real 0.8B the residual MLP ran 20/20 exact (10 digits x 2 readouts, 280 page runs), the deep MLP 20/20 (740 page runs) and the CNN 12/12 (6 digits x 2 readouts, 1320 page runs), each page's tokens equal to the numpy token golden and each final output equal to `reference.run` ([wide_residual](results/a100_wide_residual.log), [wide_deep](results/a100_wide_deep.log), [wide_cnn](results/a100_wide_cnn.log)). Those runs used the first 10 held-out digits (CNN: 6), so they show exactness, not accuracy on all 540; the machine accuracy for these three rows was not measured.

Honest reading of the losses:

* **v2.0 hardware (shift = 0) is the dominant loss.** With no shift the layer's output scale is tied to the integer weight range, so PTQ collapses
  (24-49% on everything but the tiny net, 14% on the CNN). qwenTPU's own 87.6% on the digits net is reproduced (87.59%) only by integer-aware training
  from scratch, and that does not scale: 4-layer and conv nets stay at 79-84% (MLPs) or fail (CNN). A per-layer output shift (v2.1 gadget) fixes it:
  97.6% on the deep MLP, 97.2% on the CNN. Making v2.1 real in the datapath is the single biggest accuracy lever.
* **Bias range.** The bias is one weight on a constant-1 row, so it is limited to [-16, 15] *in accumulator units*; with shift s that is only 2^-s of the activation range. This
  clipped 40-60% of the biases of the small digits net (76% PTQ vs 85% with 4 constant rows). Costs rows per tile (`Graph.bias_rows`).
* Final logits are relu+clamp (0..31): negative logits become 0 and ties are possible (`logits="linear"` quantizes signed logits for comparison but is not machine-native).
* 5-bit weights/activations, input quantization (0..31, or 0..12 for qwenTPU parity), per-tensor scales, no per-channel scales: PTQ gap to float is 0-1 point on the deep nets
  and 5-14 points on the tiny 15-15-10 net before QAT. Float numbers vary +-1.5 points with the training seed; ours are one fixed seed each.
* No BatchNorm folding yet, no signed interior activations, classification-only QAT.
