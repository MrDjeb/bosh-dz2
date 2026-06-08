"""Главный цикл обучения нейрорегулятора."""

import random
import time
from typing import Literal

import torch

from bosh_dz2.config import DEVICE, DTYPE, TrainingConfig
from bosh_dz2.controller import Controller
from bosh_dz2.export import plot_training, save_for_simulink
from bosh_dz2.loss import (
    LossComponents,
    compute_loss,
    compute_loss_sin,
    compute_loss_step,
)
from bosh_dz2.plant import build_plant
from bosh_dz2.references import ReferenceData, build_references

TeacherSample = tuple[Literal["step", "sin"], int]


def _print_header(cfg: TrainingConfig, refs: ReferenceData, mode: str) -> None:
    print(f"Обучение на CPU ({mode})")
    print(f"  Ступеньки: {refs.n_step_exp} × {cfg.Tsim_step}с ({refs.N_step} шагов)")
    print(f"  Синусы:    {refs.n_sin_exp} × {cfg.Tsim_sin}с ({refs.N_sin} шагов)")
    print(f"  Установившаяся часть: t >= {cfg.t_settle}с")
    print(
        f"  Веса: os={cfg.w_os}, set={cfg.w_set}, steady={cfg.w_steady}, "
        f"track={cfg.w_track}, ise={cfg.w_ise}"
    )
    print()


def _optimizer_step(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: LossComponents,
    max_norm: float,
) -> None:
    optimizer.zero_grad()
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
    optimizer.step()


def _record_epoch(
    loss: LossComponents,
    J_hist: list[float],
    components_hist: dict[str, list[float]],
) -> None:
    J_hist.append(loss.total.item())
    components_hist["os"].append(loss.os.item())
    components_hist["set"].append(loss.set.item())
    components_hist["steady"].append(loss.steady.item())
    components_hist["track"].append(loss.track.item())
    components_hist["ise"].append(loss.ise.item())


def _teacher_samples(refs: ReferenceData, shuffle: bool) -> list[TeacherSample]:
    samples: list[TeacherSample] = [("step", i) for i in range(refs.n_step_exp)] + [
        ("sin", i) for i in range(refs.n_sin_exp)
    ]
    if shuffle:
        random.shuffle(samples)
    return samples


def train(cfg: TrainingConfig | None = None) -> None:
    cfg = cfg or TrainingConfig()
    refs = build_references(cfg)
    plant = build_plant()
    net = Controller().to(DEVICE).to(DTYPE)
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr)

    _print_header(cfg, refs, "batch offline")

    J_hist: list[float] = []
    components_hist: dict[str, list[float]] = {
        "os": [],
        "set": [],
        "steady": [],
        "track": [],
        "ise": [],
    }

    t_start = time.time()
    for ep in range(1, cfg.n_epoch + 1):
        t_ep = time.time()

        loss = compute_loss(net, plant, refs, cfg)
        _optimizer_step(net, optimizer, loss, cfg.max_norm)
        _record_epoch(loss, J_hist, components_hist)

        dt = time.time() - t_ep
        if ep <= 5 or ep % 10 == 0:
            eta = dt * (cfg.n_epoch - ep)
            print(
                f"  Эпоха {ep:3d}/{cfg.n_epoch}  J={loss.total.item():.4f}  "
                f"[os={loss.os.item():.4f} set={loss.set.item():.4f} "
                f"steady={loss.steady.item():.5f} "
                f"track={loss.track.item():.4f} ise={loss.ise.item():.4f}]  "
                f"({dt:.2f}с, ETA {eta:.0f}с)"
            )

    print(f"\nВремя обучения: {time.time() - t_start:.1f} с")

    save_for_simulink(net, cfg.U_LIM, cfg.Ts, cfg.weights_file)
    plot_training(J_hist, components_hist, cfg.curves_file)


def train_online(cfg: TrainingConfig | None = None) -> None:
    """Online Supervised Learning: обновление весов после каждого эталона от учителя."""
    cfg = cfg or TrainingConfig()
    refs = build_references(cfg)
    plant = build_plant()
    net = Controller().to(DEVICE).to(DTYPE)
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr)

    _print_header(cfg, refs, "online supervised (OSL)")
    print(
        f"  Примеров за эпоху: {refs.n_step_exp + refs.n_sin_exp} "
        f"(обновление после каждого)"
    )
    print()

    J_hist: list[float] = []
    components_hist: dict[str, list[float]] = {
        "os": [],
        "set": [],
        "steady": [],
        "track": [],
        "ise": [],
    }

    t_start = time.time()
    for ep in range(1, cfg.n_epoch + 1):
        t_ep = time.time()

        for kind, idx in _teacher_samples(refs, shuffle=True):
            if kind == "step":
                sample_loss = compute_loss_step(net, plant, refs, cfg, idx)
            else:
                sample_loss = compute_loss_sin(net, plant, refs, cfg, idx)
            _optimizer_step(net, optimizer, sample_loss, cfg.max_norm)

        with torch.no_grad():
            epoch_loss = compute_loss(net, plant, refs, cfg)
        _record_epoch(epoch_loss, J_hist, components_hist)

        dt = time.time() - t_ep
        if ep <= 5 or ep % 10 == 0:
            eta = dt * (cfg.n_epoch - ep)
            print(
                f"  Эпоха {ep:3d}/{cfg.n_epoch}  J={epoch_loss.total.item():.4f}  "
                f"[os={epoch_loss.os.item():.4f} set={epoch_loss.set.item():.4f} "
                f"steady={epoch_loss.steady.item():.5f} "
                f"track={epoch_loss.track.item():.4f} ise={epoch_loss.ise.item():.4f}]  "
                f"({dt:.2f}с, ETA {eta:.0f}с)"
            )

    print(f"\nВремя обучения: {time.time() - t_start:.1f} с")

    save_for_simulink(net, cfg.U_LIM, cfg.Ts, cfg.online_weights_file)
    plot_training(J_hist, components_hist, cfg.online_curves_file)

    print(f"\nВремя обучения: {time.time() - t_start:.1f} с")

    save_for_simulink(net, cfg.U_LIM, cfg.Ts, cfg.online_weights_file)
    plot_training(J_hist, components_hist, cfg.online_curves_file)
