"""Модель объекта управления (привод)."""

import numpy as np
import torch
from scipy import signal

from bosh_dz2.config import DEVICE, DTYPE

# Передаточная функция привода
# W(s) = 2.55 / (s^2 + 0.0473s + 0.0141)

def build_plant():
    num = [2.55]
    den = np.convolve([1, 0], np.convolve([0.0332, 1], [0.0141, 1]))
    A, B, C, _ = signal.tf2ss(num, den)
    return (
        torch.tensor(A, dtype=DTYPE, device=DEVICE),
        torch.tensor(B, dtype=DTYPE, device=DEVICE),
        torch.tensor(C, dtype=DTYPE, device=DEVICE),
    )
