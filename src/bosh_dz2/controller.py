"""Нейросетевой регулятор."""

import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, n_in: int = 4, n_hid: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, 1)

        torch.manual_seed(1)
        with torch.no_grad():
            self.fc1.weight.normal_(0, 0.5)
            self.fc1.bias.normal_(0, 0.1)
            self.fc2.weight.normal_(0, 0.1)
            self.fc2.bias.zero_()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(inp))
        return self.fc2(h).squeeze(-1)
