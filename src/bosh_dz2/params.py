"""Упаковка и распаковка весов нейросети для SciPy."""

import numpy as np
import torch
import torch.nn as nn

from bosh_dz2.config import DEVICE, DTYPE


def pack_params(net: nn.Module) -> np.ndarray:
    return np.concatenate([
        p.detach().cpu().numpy().astype(np.float64).ravel()
        for p in net.parameters()
    ])


def unpack_params(net: nn.Module, w: np.ndarray) -> None:
    w = np.asarray(w, dtype=np.float64)
    offset = 0
    for p in net.parameters():
        n = p.numel()
        chunk = torch.tensor(w[offset:offset + n], dtype=DTYPE, device=DEVICE)
        p.data.copy_(chunk.view_as(p))
        offset += n
