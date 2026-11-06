window.RESULTS = {
 "_note": "Every row is traced to a log in results/. null = not measured; the page shows 'pending'. Run `make site` after editing.",
 "headline": {
  "networks_run": 12,
  "tokens_replayed": null,
  "mismatches": 0,
  "qwen_models": 3
 },
 "columns": [
  {
   "key": "network",
   "label": "Network"
  },
  {
   "key": "float",
   "label": "Float (torch)"
  },
  {
   "key": "int_v20",
   "label": "Integer v2.0, PTQ / +QAT"
  },
  {
   "key": "int_v21",
   "label": "Integer v2.1, 4 bias rows, PTQ / +QAT"
  },
  {
   "key": "sim",
   "label": "Machine (CPU simulator) vs reference"
  },
  {
   "key": "hardware",
   "label": "Real Qwen3.5"
  }
 ],
 "rows": [
  {
   "network": "digits MLP 15-15-10",
   "float": "90.4%",
   "int_v20": "43.7% / 82.6%",
   "int_v21": "85.2% / 89.8%",
   "sim": "bit-exact: 540/540 digits (CPU simulator), plus tests at shifts 2 and 4",
   "hardware": "A100 bf16, 0.8B: 40/40 exact (9-layer machine, 20 digits x 2 readouts); 6/6 (14-layer machine). Also 40/40 on 2B and on 4B. Vectorized machine, 1 and 2 lanes: 40/40 each"
  },
  {
   "network": "deep MLP 64-96-64-32-10",
   "float": "97.8%",
   "int_v20": "24.1% / 78.9%",
   "int_v21": "97.8% / 97.8%",
   "sim": "in 37 pages (tile groups for the 64- and 96-wide layers): token golden 540/540 digits; circuit simulator 1 digit",
   "hardware": "A100 bf16, 0.8B: 20/20 exact in 37 pages (10 digits x 2 readouts, 740 page runs, 0 mismatches)"
  },
  {
   "network": "CNN conv4-conv8-avgpool-fc",
   "float": "97.4%",
   "int_v20": "14.3% / 14.3%",
   "int_v21": "96.5% / 97.0%",
   "sim": "in 110 pages (512-wide Linear in tile groups): token golden 540/540 digits; circuit simulator 1 digit",
   "hardware": "A100 bf16, 0.8B: 12/12 exact in 110 pages (6 digits x 2 readouts, 1,320 page runs, 0 mismatches)"
  },
  {
   "network": "residual MLP 64-32-[32+res]-10",
   "float": "97.4%",
   "int_v20": "49.4% / 83.7%",
   "int_v21": "96.9% / 96.9%",
   "sim": "in 14 pages (tile groups): token golden 540/540 digits; circuit simulator 1 digit",
   "hardware": "A100 bf16, 0.8B: 20/20 exact in 14 pages (10 digits x 2 readouts, 280 page runs, 0 mismatches)"
  },
  {
   "network": "random MLP 20-8-4 (K=20 spans 2 tiles)",
   "float": "n/a (random weights)",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "random-weight program; same program checked on the A100",
   "hardware": "A100: 6/6 exact on 0.8B, on 2B and on 4B"
  },
  {
   "network": "random 4-layer MLP 8-5-5-5-4",
   "float": "n/a (random weights)",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "random-weight program; same program checked on the A100",
   "hardware": "A100: 6/6 exact on 0.8B, on 2B and on 4B"
  },
  {
   "network": "residual block (concat + identity rows)",
   "float": "n/a (random weights)",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "random-weight program; same program checked on the A100",
   "hardware": "A100: 6/6 exact on 0.8B, on 2B and on 4B"
  },
  {
   "network": "residual MLP 15-10-[10+res]-10",
   "float": "n/a",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "examples/run_residual.py: machine equals reference.run",
   "hardware": "A100 bf16, 0.8B: 40/40 exact (20 digits x 2 readouts)"
  },
  {
   "network": "paged residual MLP 44-20-[20+res]-20-10 (70 neurons)",
   "float": "97.2%",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "7 pages: token golden 540/540 digits; circuit simulator 1 digit",
   "hardware": "A100 bf16, 0.8B: 20/20 exact (10 digits x 2 readouts, 140 page runs)"
  },
  {
   "network": "paged deep MLP 44-44-44-44-10 (142 neurons)",
   "float": "97.2%",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "10 pages: token golden 540/540 digits; circuit simulator 3 digits",
   "hardware": "A100 bf16, 0.8B: 20/20 exact (200 page runs)"
  },
  {
   "network": "paged strided CNN 1x8x8 (426 neurons)",
   "float": "96.9%",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "36 pages: token golden 540/540 digits; circuit simulator 1 digit",
   "hardware": "A100 bf16, 0.8B: 20/20 exact (720 page runs)"
  },
  {
   "network": "784-8-4, random weights and inputs (784-wide first layer)",
   "float": "n/a (random weights)",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "checked only against reference.run",
   "hardware": "A100 bf16, 0.8B: 6/6 exact in 9 pages (3 inputs x 2 readouts, 54 page runs); shows exactness, not accuracy"
  },
  {
   "network": "126-neuron machine, random weights (all 126 columns)",
   "float": "n/a (random weights)",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "exact circuit evaluator",
   "hardware": "A100 bf16, 0.8B: 6/6 exact; a depth-126 chain also 6/6 (peak 33.4 GB)"
  }
 ],
 "ops": [
  {
   "op": "Linear / matmul",
   "status": "yes",
   "note": "Linear->Linear with no ReLU between merges into one matvec"
  },
  {
   "op": "ReLU",
   "status": "yes",
   "note": "fused relu_clamp of the producing matvec: clip(acc >> shift, 0, 31)"
  },
  {
   "op": "Other activations (GELU, SiLU, tanh, sigmoid)",
   "status": "no",
   "note": "rejected; a 32-entry table as a ReLU basis could cover them (DESIGN.md section 10)"
  },
  {
   "op": "Add / residual",
   "status": "yes",
   "note": "identity rows in the consuming matvec over a concat; flat features only"
  },
  {
   "op": "Conv2d",
   "status": "yes",
   "note": "groups=1, dilation=1, zero padding, any stride; im2col gather + grouped matvec; costs P times the neurons"
  },
  {
   "op": "Pooling",
   "status": "yes",
   "note": "AvgPool2d / AdaptiveAvgPool2d supported; MaxPool2d rejected (max(a,b) = a + relu(b-a) not implemented)"
  },
  {
   "op": "Normalization (LayerNorm, BatchNorm)",
   "status": "no",
   "note": "rejected; eval BatchNorm could be folded into the previous layer, not automatic yet"
  },
  {
   "op": "Softmax / attention",
   "status": "no",
   "note": "rejected; a trailing softmax before argmax is dropped, otherwise needs exp/reciprocal tables and data x data products"
  },
  {
   "op": "Embedding",
   "status": "yes",
   "note": "on the integer input only, as a one-hot matvec"
  },
  {
   "op": "Recurrent layers (RNN, LSTM, GRU)",
   "status": "no",
   "note": "not supported; no loops in the machine"
  },
  {
   "op": "Data-dependent control flow",
   "status": "no",
   "note": "no branches, no loops, no early exit"
  },
  {
   "op": "Floating-point tensors at runtime",
   "status": "no",
   "note": "the machine is integer; the graph is quantized first"
  }
 ],
 "headline_note": "networks_run = 12 distinct networks bit-exact on the real Qwen3.5-0.8B (A100): digits MLP, K=20 two-tile MLP, 4-layer MLP, residual block, residual MLP 15-10-[10+res]-10, three paged networks (residual 44-20, deep 44x4, strided CNN), the three wide originals (residual 64-32, deep 64-96-64-32-10, CNN with a 512-wide Linear) and a 784-wide first layer (results/a100_digits.log, a100_general.log, a100_residual_q08.log, a100_stream.log, a100_wide_*.log, a100_wide784.log). qwen_models = 3: the digits MLP and the four-program model also ran exact on 2B and 4B (results/a100_*_q2.log, a100_*_q4.log). tokens_replayed is not reported."
}
;
