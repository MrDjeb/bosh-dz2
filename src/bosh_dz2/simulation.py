"""Симуляция замкнутого контура."""

import torch
import torch.nn as nn

from bosh_dz2.config import DEVICE, DTYPE


def simulate(
    net: nn.Module,
    refs: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    Ts: float,
    U_LIM: float,
) -> torch.Tensor:
    """refs: (N, M) -> Y: (N, M)"""
    N, M = refs.shape
    nx = A.shape[0]

    x = torch.zeros(nx, M, dtype=DTYPE, device=DEVICE)
    sum_e = torch.zeros(M, dtype=DTYPE, device=DEVICE)
    e_km1 = torch.zeros(M, dtype=DTYPE, device=DEVICE)
    e_km2 = torch.zeros(M, dtype=DTYPE, device=DEVICE)

    Y = []
    for k in range(N):
        y = (C @ x).squeeze(0)
        e = refs[k] - y

        inp = torch.stack([e, e_km1, e_km2, sum_e], dim=1)
        u_raw = net(inp)
        u = U_LIM * torch.tanh(u_raw / U_LIM)

        Bu = B @ u.unsqueeze(0)
        x = x + Ts * (A @ x + Bu)
        sum_e = sum_e + e * Ts

        e_km2 = e_km1
        e_km1 = e

        Y.append(y)

    return torch.stack(Y, dim=0)
