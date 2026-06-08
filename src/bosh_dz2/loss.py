"""Функционал качества нейрорегулятора."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from bosh_dz2.config import TrainingConfig
from bosh_dz2.references import ReferenceData
from bosh_dz2.simulation import simulate


@dataclass(frozen=True)
class LossComponents:
    total: torch.Tensor
    os: torch.Tensor
    set: torch.Tensor
    steady: torch.Tensor
    track: torch.Tensor
    ise: torch.Tensor


def compute_loss(
    net: nn.Module,
    plant: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    refs: ReferenceData,
    cfg: TrainingConfig,
) -> LossComponents:
    A, B, C = plant

    Y_step = simulate(net, refs.step_refs, A, B, C, cfg.Ts, cfg.U_LIM)
    e_step = refs.step_refs - Y_step

    J_os_total = 0.0
    for i in range(refs.n_step_exp):
        sgn = torch.sign(refs.step_ref_consts[i])
        z = torch.clamp(sgn * (Y_step[:, i] - refs.step_ref_consts[i]), min=0.0)
        M_os = torch.logsumexp(20.0 * z, dim=0) / 20.0
        J_os_total = J_os_total + M_os ** 2
    J_os = J_os_total / refs.n_step_exp

    viol = torch.clamp(torch.abs(e_step) - refs.deltas.unsqueeze(0), min=0.0)
    J_set = (refs.t_step.unsqueeze(1) * viol ** 2).sum() * cfg.Ts / refs.n_step_exp

    rel_err_step = e_step / refs.step_ref_consts.unsqueeze(0)
    J_steady = (
        rel_err_step ** 2 * refs.mask_step_steady
    ).sum() * cfg.Ts / refs.n_step_exp

    J_ise_step = (e_step ** 2).sum() * cfg.Ts / refs.n_step_exp

    Y_sin = simulate(net, refs.sin_refs, A, B, C, cfg.Ts, cfg.U_LIM)
    e_sin = refs.sin_refs - Y_sin

    J_track = (
        e_sin ** 2 * refs.mask_sin_steady
    ).sum() * cfg.Ts / refs.n_sin_exp
    J_ise_sin = (e_sin ** 2).sum() * cfg.Ts / refs.n_sin_exp
    J_ise = (J_ise_step + J_ise_sin) / 2

    J = (
        cfg.w_os * J_os
        + cfg.w_set * J_set
        + cfg.w_steady * J_steady
        + cfg.w_track * J_track
        + cfg.w_ise * J_ise
    )

    return LossComponents(
        total=J,
        os=J_os,
        set=J_set,
        steady=J_steady,
        track=J_track,
        ise=J_ise,
    )
