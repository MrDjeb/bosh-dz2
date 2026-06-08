"""Общие настройки выполнения и гиперпараметры обучения."""

from dataclasses import dataclass

import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float32
torch.set_num_threads(4)


@dataclass(frozen=True)
class TrainingConfig:
    Ts: float = 0.001  # Время дискретизации
    U_LIM: float = 1.0  # Ограничение на амплитуду управления
    Tsim_step: float = 5.0  # Длительность сигнала ступеньки
    Tsim_sin: float = 10.0  # Длительность сигнала синуса
    t_settle: float = 5.0  # Время установления
    n_epoch: int = 200  # Количество эпох обучения
    lr: float = 0.1  # Скорость обучения
    max_norm: float = 5.0  # Ограничение на норму градиента
    w_os: float = 5.0  # Вес компоненты перерегулирования
    w_set: float = 2.0
    w_steady: float = 50.0
    w_track: float = 3.0
    w_ise: float = 0.5
    band: float = 0.02
    weights_file: str = "nn_weights.mat"
    curves_file: str = "training_curves.png"
    online_weights_file: str = "nn_weights_online.mat"
    online_curves_file: str = "training_curves_online.png"
    online_lm_weights_file: str = "nn_weights_online_lm.mat"
    online_lm_curves_file: str = "training_curves_online_lm.png"
    lm_max_nfev: int | None = 50
    lm_ftol: float = 1e-8
    lm_xtol: float = 1e-8
    lm_gtol: float = 1e-8
