"""Векторы невязок для Levenberg–Marquardt: J = ||r||²."""

import numpy as np
import torch
import torch.nn as nn

from bosh_dz2.config import DTYPE, TrainingConfig
from bosh_dz2.references import ReferenceData
from bosh_dz2.simulation import simulate


def residuals_step(
    net: nn.Module,
    plant: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    refs: ReferenceData,
    cfg: TrainingConfig,
    idx: int,
) -> np.ndarray:
    """Невязки одной ступеньки; sum(r²) = compute_loss_step(...).total."""
    A, B, C = plant
    ref = refs.step_refs[:, idx:idx + 1]

    with torch.no_grad():
        Y = simulate(net, ref, A, B, C, cfg.Ts, cfg.U_LIM)
        e = ref - Y

        sgn = torch.sign(refs.step_ref_consts[idx])
        z = torch.clamp(sgn * (Y[:, 0] - refs.step_ref_consts[idx]), min=0.0)
        M_os = torch.logsumexp(20.0 * z, dim=0) / 20.0
        r_os = torch.sqrt(torch.tensor(cfg.w_os, dtype=DTYPE)) * M_os

        viol = torch.clamp(torch.abs(e) - refs.deltas[idx], min=0.0)
        scale_set = torch.sqrt(cfg.w_set * refs.t_step * cfg.Ts)
        r_set = (scale_set * viol.squeeze(1)).reshape(-1)

        rel_err = e / refs.step_ref_consts[idx]
        scale_steady = torch.sqrt(cfg.w_steady * refs.mask_step_steady * cfg.Ts)
        r_steady = (scale_steady * rel_err).reshape(-1)

        scale_ise = torch.sqrt(torch.tensor(cfg.w_ise * cfg.Ts, dtype=DTYPE))
        r_ise = (scale_ise * e.squeeze(1)).reshape(-1)

        parts = [r_os.reshape(1), r_set, r_steady, r_ise]

    return torch.cat(parts).cpu().numpy().astype(np.float64)


def residuals_sin(
    net: nn.Module,
    plant: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    refs: ReferenceData,
    cfg: TrainingConfig,
    idx: int,
) -> np.ndarray:
    """Невязки одного синуса; sum(r²) = compute_loss_sin(...).total."""
    A, B, C = plant
    ref = refs.sin_refs[:, idx:idx + 1]

    with torch.no_grad():
        Y = simulate(net, ref, A, B, C, cfg.Ts, cfg.U_LIM)
        e = ref - Y

        scale_track = torch.sqrt(cfg.w_track * refs.mask_sin_steady * cfg.Ts)
        r_track = (scale_track * e).reshape(-1)

        scale_ise = torch.sqrt(torch.tensor(cfg.w_ise * cfg.Ts, dtype=DTYPE))
        r_ise = (scale_ise * e.squeeze(1)).reshape(-1)

    return torch.cat([r_track, r_ise]).cpu().numpy().astype(np.float64)
