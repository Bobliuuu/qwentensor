Held-out sklearn digits (540). Machine accuracy is only measured for the digits MLP.

| network | neurons | fits the machine | machine == `reference.run` | float / int / machine accuracy |
|---|---|---|---|---|
| digits MLP 15-15-10 (496-token prompt) | 25 | yes | **540/540 digits** (about 4.1 s/digit under load) | 0.9037 / 0.8944 / 0.8944 |
| **paged, wide** deep MLP 64-96-64-32-10 (the full network above, `examples/run_wide.py deep`) | 202 | yes, 37 pages on a 41-column machine (neurons read 64 and 96 inputs, tile groups) | token golden 540/540; circuit simulator 1 digit | 0.9778 / 0.9759 / not measured |
| **paged, wide** residual MLP 64-32-[32+res]-10 (`... residual`) | 106 (408 columns) | yes, 14 pages on a 43-column machine | token golden 540/540; circuit simulator 1 digit | 0.9741 / 0.9611 / not measured |
| **paged, wide** CNN conv4-conv8-avgpool-fc (`... cnn`) | 778 | yes, 110 pages on a 58-column machine (AvgPool folds into a 512-wide Linear: 31 tiles, 11 groups) | token golden 540/540; circuit simulator 1 digit | 0.9741 / 0.9722 / not measured |
| deep variant 15-24-16-10 | 50 | yes | first 3 digits only | 0.9056 / 0.9037 / not measured |
| residual variant, width 10 on the 15 pooled features | 40 | yes | first 3 digits only | 0.8685 / 0.8426 / not measured |
| CNN variant 4x4 input, conv3x3 1->2 ch + fc | 42 | yes | first 3 digits only | 0.9185 / 0.8722 / not measured |
| **paged** deep MLP 44-44-44-44-10 (`examples/run_stream.py deep`) | 142 | yes, 10 pages on a 45-column machine | token golden 540/540; circuit simulator 3 digits | 0.9722 / 0.9648 / not measured |
| **paged** residual MLP 44-20-[20+res]-20-10 (`... residual`) | 70 | yes, 7 pages on a 44-column machine | token golden 540/540; circuit simulator 1 digit | 0.9722 / 0.9704 / not measured |
| **paged** strided CNN 1x8x8 (`... cnn`) | 426 | yes, 36 pages on a 42-column machine | token golden 540/540; circuit simulator 1 digit | 0.9685 / 0.9574 / not measured |

The variants' float / int accuracy is from `reference.run` on all 540 digits; their machine accuracy was not measured. "Token golden" is the independent numpy model of the machine
(`asm.reference_emit`), run page by page. All of the above is the CPU simulator. Real-model claims are only the A100 logs in Status.
