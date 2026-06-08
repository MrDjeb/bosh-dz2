"""Эталонные сигналы и маски для обучения."""

from dataclasses import dataclass

import numpy as np
import torch

from bosh_dz2.config import DEVICE, DTYPE, TrainingConfig


@dataclass(frozen=True)
class ReferenceData:
    step_refs: torch.Tensor
    sin_refs: torch.Tensor
    step_ref_consts: torch.Tensor
    t_step: torch.Tensor
    t_sin: torch.Tensor
    mask_step_steady: torch.Tensor
    mask_sin_steady: torch.Tensor
    deltas: torch.Tensor
    n_step_exp: int
    n_sin_exp: int
    N_step: int
    N_sin: int


def build_references(cfg: TrainingConfig) -> ReferenceData:
    N_step = int(cfg.Tsim_step / cfg.Ts)
    N_sin = int(cfg.Tsim_sin / cfg.Ts)

    t_step = torch.arange(N_step, dtype=DTYPE, device=DEVICE) * cfg.Ts
    t_sin = torch.arange(N_sin, dtype=DTYPE, device=DEVICE) * cfg.Ts

    step_amps_np = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]
    n_step_exp = len(step_amps_np)
    step_refs = torch.tensor(
        [[a] * N_step for a in step_amps_np],
        dtype=DTYPE,
        device=DEVICE,
    ).T
    step_ref_consts = torch.tensor(step_amps_np, dtype=DTYPE, device=DEVICE)

    sin_specs = [
        (0.4, 0.0, 0.3),
        (0.6, 0.0, 0.4),
        (0.7, 0.0, 0.5),
        (0.5, 0.0, 0.7),
    ]
    n_sin_exp = len(sin_specs)
    sin_refs = torch.stack([
        amp * torch.sin(2 * np.pi * freq * t_sin) + bias
        for amp, bias, freq in sin_specs
    ], dim=1)

    mask_step_steady = (t_step >= cfg.t_settle).float().unsqueeze(1)
    mask_sin_steady = (t_sin >= cfg.t_settle).float().unsqueeze(1)
    deltas = cfg.band * step_ref_consts

    return ReferenceData(
        step_refs=step_refs,
        sin_refs=sin_refs,
        step_ref_consts=step_ref_consts,
        t_step=t_step,
        t_sin=t_sin,
        mask_step_steady=mask_step_steady,
        mask_sin_steady=mask_sin_steady,
        deltas=deltas,
        n_step_exp=n_step_exp,
        n_sin_exp=n_sin_exp,
        N_step=N_step,
        N_sin=N_sin,
    )
