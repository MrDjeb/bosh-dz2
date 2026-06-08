"""Сохранение весов и визуализация обучения."""

import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.io import savemat


def save_for_simulink(net: nn.Module, U_LIM: float, Ts: float, filename: str) -> None:
    nn_dict = {
        'W1': net.fc1.weight.detach().cpu().double().numpy(),
        'b1': net.fc1.bias.detach().cpu().double().numpy().reshape(-1, 1),
        'W2': net.fc2.weight.detach().cpu().double().numpy(),
        'b2': float(net.fc2.bias.detach().cpu().double().numpy()[0]),
        'U_LIM': float(U_LIM),
        'Ts': float(Ts),
    }
    savemat(filename, {'nn': nn_dict})
    print(f"\nВеса сохранены в {filename}")
    print("В MATLAB:  load('nn_weights.mat');")


def plot_training(J_hist: list[float], comps: dict[str, list[float]], filename: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].plot(J_hist)
    axes[0, 0].set_title('J полное')
    axes[0, 1].plot(comps['os'])
    axes[0, 1].set_title('Перерегулирование (step)')
    axes[0, 2].plot(comps['set'])
    axes[0, 2].set_title('Время рег-я (step)')
    axes[1, 0].plot(comps['steady'])
    axes[1, 0].set_title('Уст. ошибка step')
    axes[1, 1].plot(comps['track'])
    axes[1, 1].set_title('Уст. ошибка sin')
    axes[1, 2].plot(comps['ise'])
    axes[1, 2].set_title('ISE общий')
    for ax in axes.flat:
        if ax.has_data():
            ax.grid(True)
            ax.set_xlabel('Эпоха')
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.show()
