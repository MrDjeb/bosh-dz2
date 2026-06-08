"""Общие настройки выполнения и гиперпараметры обучения."""

from dataclasses import dataclass

import torch

DEVICE = torch.device('cpu')
DTYPE = torch.float32
torch.set_num_threads(4)


@dataclass(frozen=True)
class TrainingConfig:
    Ts: float = 0.001
    U_LIM: float = 1.0
    Tsim_step: float = 5.0
    Tsim_sin: float = 10.0
    t_settle: float = 5.0
    n_epoch: int = 200
    lr: float = 0.1
    max_norm: float = 5.0
    w_os: float = 5.0
    w_set: float = 2.0
    w_steady: float = 50.0
    w_track: float = 3.0
    w_ise: float = 0.5
    band: float = 0.02
    weights_file: str = 'nn_weights.mat'
    curves_file: str = 'training_curves.png'
