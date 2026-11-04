window.RESULTS = {
 "_note": "Verified rows only. null = not measured yet; the page shows 'pending'. Run `make site` after editing.",
 "headline": {
  "networks_run": 4,
  "tokens_replayed": null,
  "mismatches": 0,
  "qwen_models": 1
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
   "hardware": "Qwen3.5-0.8B, A100 bf16: 40/40 exact (9-layer machine, 20 digits x 2 readouts); 6/6 exact (14-layer machine)"
  },
  {
   "network": "deep MLP 64-96-64-32-10",
   "float": "97.8%",
   "int_v20": "24.1% / 78.9%",
   "int_v21": "97.8% / 97.8%",
   "sim": "not run (202 neurons: DoesNotFit); a 15-24-16-10 variant matched on 3 digits",
   "hardware": "not run yet"
  },
  {
   "network": "CNN conv4-conv8-avgpool-fc",
   "float": "97.4%",
   "int_v20": "14.3% / 14.3%",
   "int_v21": "96.5% / 97.0%",
   "sim": "not run (778 neurons: DoesNotFit); a small 4x4 variant matched on 3 digits",
   "hardware": "not run yet"
  },
  {
   "network": "residual MLP 64-32-[32+res]-10",
   "float": "97.4%",
   "int_v20": "49.4% / 83.7%",
   "int_v21": "96.9% / 96.9%",
   "sim": "not run (106 neurons, 408 columns: DoesNotFit); a width-10 variant matched on 3 digits",
   "hardware": "not run (a residual block 6/6 exact on A100; this network not run)"
  },
  {
   "network": "random MLP 20-8-4 (K=20 spans 2 tiles)",
   "float": "n/a (random weights)",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "random-weight program; same program checked on the A100",
   "hardware": "A100: 6/6 exact"
  },
  {
   "network": "random 4-layer MLP 8-5-5-5-4",
   "float": "n/a (random weights)",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "random-weight program; same program checked on the A100",
   "hardware": "A100: 6/6 exact"
  },
  {
   "network": "residual block (concat + identity rows)",
   "float": "n/a (random weights)",
   "int_v20": "n/a",
   "int_v21": "n/a",
   "sim": "random-weight program; same program checked on the A100",
   "hardware": "A100: 6/6 exact"
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
 "headline_note": "networks_run: programs bit-exact on the real Qwen3.5-0.8B (A100): digits MLP (9-layer machine, 40/40, results/a100_digits.log; and in the 14-layer machine), K=20 two-tile MLP, 4-layer MLP, residual block (14-layer machine, 24/24, results/a100_general.log). qwen_models: 1 (0.8B). tokens_replayed not reported."
};
