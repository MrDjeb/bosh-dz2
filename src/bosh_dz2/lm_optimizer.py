"""Levenberg–Marquardt через scipy.optimize.least_squares."""

from collections.abc import Callable

import numpy as np
import torch.nn as nn
from scipy.optimize import OptimizeResult, least_squares

from bosh_dz2.config import TrainingConfig
from bosh_dz2.params import pack_params, unpack_params


def levenberg_marquardt_step(
    net: nn.Module,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    cfg: TrainingConfig,
) -> OptimizeResult:
    """Минимизация ||r(w)||² методом LM (MINPACK)."""
    w0 = pack_params(net)
    result = least_squares(
        residual_fn,
        w0,
        method='lm',
        ftol=cfg.lm_ftol,
        xtol=cfg.lm_xtol,
        gtol=cfg.lm_gtol,
        max_nfev=cfg.lm_max_nfev,
    )
    unpack_params(net, result.x)
    return result
