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

import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bosh_dz2.training import train

if __name__ == '__main__':
    train()
