"""Small CNN on 8x8 digits: conv3x3(1->4,pad1)+ReLU, conv3x3(4->8,pad1)+ReLU, AvgPool2d(2), Flatten, Linear(128->10).
Conv = im2col gather + grouped matvec (64 positions share one 9x4 / 36x8 weight tile); the AvgPool is folded into the Linear (K = 512)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from common import digits, train_float, report
from qwentensor.quant import QConfig


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 4, 3, padding=1); self.c2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.AvgPool2d(2); self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        x = torch.relu(self.c1(x)); x = torch.relu(self.c2(x)); x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


Xtr, Xte, ytr, yte = digits("raw64")
torch.manual_seed(0)
model = CNN()
train_float(model, Xtr, ytr, epochs=60, shape=(1, 8, 8))
if __name__ == "__main__":
    q = {"QAT from PTQ (20 ep)": dict(epochs=20, lr=0.01)}
    report("CNN 2conv+pool", model, (1, 8, 8), Xtr, ytr, Xte, yte, {
        "v2.0 no-shift, bias_rows=1": (QConfig("v2.0"), q),
        "v2.1 shift, bias_rows=1": (QConfig("v2.1"), q),
        "v2.1 shift, bias_rows=4": (QConfig("v2.1", bias_rows=4), q),
    })
