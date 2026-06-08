# bosh-dz2

```
poetry install
poetry run python -m bosh_dz2
```

```
src/bosh_dz2/
├── __init__.py       # публичный API пакета
├── __main__.py       # python -m bosh_dz2
├── main.py           # тонкая обёртка (обратная совместимость)
├── config.py         # DEVICE, DTYPE, TrainingConfig
├── plant.py          # модель привода
├── controller.py     # нейросеть Controller
├── simulation.py     # симуляция замкнутого контура
├── references.py     # эталонные сигналы и маски
├── loss.py           # функционал качества
├── export.py         # сохранение весов и графики
└── training.py       # цикл обучения
```