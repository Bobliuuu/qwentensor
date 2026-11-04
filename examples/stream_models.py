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


class StridedCNN(nn.Module):
    """conv3x3(1->4, pad 1) -> conv3x3(4->8, stride 2) -> conv2x2(8->8, stride 2) -> Linear(32->10), all ReLU: 426 neurons, every neuron reads <= 36
    inputs (a strided conv instead of AvgPool + a 512-wide Linear, which no neuron could read)."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 4, 3, padding=1); self.c2 = nn.Conv2d(4, 8, 3, stride=2, padding=1); self.c3 = nn.Conv2d(8, 8, 2, stride=2)
        self.fc = nn.Linear(8 * 2 * 2, 10)

    def forward(self, x):
        x = torch.relu(self.c1(x)); x = torch.relu(self.c2(x)); x = torch.relu(self.c3(x))
        return self.fc(torch.flatten(x, 1))


def cnn():
    """-> (model, Xtr64, Xte64, ytr, yte): the strided CNN on all 64 pixels (as (1, 8, 8) images)."""
    xtr, xte, ytr_, yte_ = digits("raw64")
    torch.manual_seed(0)
    return train_float(StridedCNN(), xtr, ytr_, epochs=60, shape=(1, 8, 8)), xtr, xte, ytr_, yte_
