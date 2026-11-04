"""Two networks that need paged execution (docs/STREAMING.md): more neurons than one machine, every neuron reads at most 45 inputs.

  deep MLP      44-44-44-44-10      132 neurons   (> 128 op slots; about 400 physical columns, one machine holds at most 126)
  residual MLP  44-20-[20+res]-20-10  70 neurons  (the residual add reads concat[relu(fc1(h)), h] = 40 inputs)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from common import digits, train_float

Xtr, Xte, ytr, yte = digits("top44")


def deep():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(44, 44), nn.ReLU(), nn.Linear(44, 44), nn.ReLU(), nn.Linear(44, 44), nn.ReLU(), nn.Linear(44, 10))
    return train_float(m, Xtr, ytr, epochs=150)


class ResMLP(nn.Module):
    def __init__(self, w=20):
        super().__init__()
        self.inp = nn.Linear(44, w); self.fc1 = nn.Linear(w, w); self.fc2 = nn.Linear(w, w); self.out = nn.Linear(w, 10)

    def forward(self, x):
        h = torch.relu(self.inp(x))
        h = torch.relu(h + self.fc2(torch.relu(self.fc1(h))))
        return self.out(h)


def residual():
    torch.manual_seed(0)
    return train_float(ResMLP(), Xtr, ytr, epochs=150)
