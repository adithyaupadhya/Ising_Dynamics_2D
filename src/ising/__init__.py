"""Fixed-minus boundary Ising simulation."""
from .model import IsingModel, BETA_C
from .simulator import Simulator

from .coupling import MonotoneCoupling

__all__ = ["IsingModel", "Simulator", "BETA_C", "MonotoneCoupling"]
