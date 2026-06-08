"""Главный цикл обучения нейрорегулятора."""

import time

import torch

from bosh_dz2.config import DEVICE, DTYPE, TrainingConfig
from bosh_dz2.controller import Controller
from bosh_dz2.export import plot_training, save_for_simulink
from bosh_dz2.loss import compute_loss
from bosh_dz2.plant import build_plant
from bosh_dz2.references import build_references


def train(cfg: TrainingConfig | None = None) -> None:
    cfg = cfg or TrainingConfig()
    refs = build_references(cfg)
    plant = build_plant()
    net = Controller().to(DEVICE).to(DTYPE)
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr)

    print("Обучение на CPU")
    print(f"  Ступеньки: {refs.n_step_exp} × {cfg.Tsim_step}с ({refs.N_step} шагов)")
    print(f"  Синусы:    {refs.n_sin_exp} × {cfg.Tsim_sin}с ({refs.N_sin} шагов)")
    print(f"  Установившаяся часть: t >= {cfg.t_settle}с")
    print(
        f"  Веса: os={cfg.w_os}, set={cfg.w_set}, steady={cfg.w_steady}, "
        f"track={cfg.w_track}, ise={cfg.w_ise}"
    )
    print()

    J_hist: list[float] = []
    components_hist: dict[str, list[float]] = {
        'os': [], 'set': [], 'steady': [], 'track': [], 'ise': [],
    }

    t_start = time.time()
    for ep in range(1, cfg.n_epoch + 1):
        t_ep = time.time()

        optimizer.zero_grad()
        loss = compute_loss(net, plant, refs, cfg)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_norm)
        optimizer.step()

        J_hist.append(loss.total.item())
        components_hist['os'].append(loss.os.item())
        components_hist['set'].append(loss.set.item())
        components_hist['steady'].append(loss.steady.item())
        components_hist['track'].append(loss.track.item())
        components_hist['ise'].append(loss.ise.item())

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
