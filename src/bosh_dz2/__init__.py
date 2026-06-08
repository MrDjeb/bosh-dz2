"""Нейрорегулятор для привода (ЛР7)."""

from bosh_dz2.config import DEVICE, DTYPE, TrainingConfig
from bosh_dz2.controller import Controller
from bosh_dz2.loss import LossComponents, compute_loss
from bosh_dz2.plant import build_plant
from bosh_dz2.references import ReferenceData, build_references
from bosh_dz2.simulation import simulate
from bosh_dz2.training import train

__all__ = [
    'DEVICE',
    'DTYPE',
    'Controller',
    'LossComponents',
    'ReferenceData',
    'TrainingConfig',
    'build_plant',
    'build_references',
    'compute_loss',
    'simulate',
    'train',
]
