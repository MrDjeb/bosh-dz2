from bosh_dz2.config import TrainingConfig
from bosh_dz2.training import train_online

"""Точка входа: python -m bosh_dz2"""

"""
ЛР7. Нейрорегулятор для привода (PyTorch).

Стратегия:
  - CPU (для маленькой сети быстрее GPU)
  - Два батча: ступеньки (5 с) и синусы (10 с)
  - Установившаяся ошибка считается с t >= 5 с

Функционал качества:
    J = w_os    * softOS_step          (перерегулирование)
      + w_set   * ITSE_outside_step    (время регулирования)
      + w_steady* steady_step          (установившаяся ошибка ступенек)
      + w_track * ISE_steady_sin       (установившаяся ошибка слежения синусов)
      + w_ise   * ISE_all              (стабилизатор)
"""


def main() -> None:
    train_online(cfg=TrainingConfig(n_epoch=2))


if __name__ == "__main__":
    main()
    main()
