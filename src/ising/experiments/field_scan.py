"""Finite-time field comparisons, not equilibrium estimates."""
import numpy as np
from ising import IsingModel
from ising.observables import largest_plus_fraction
from .trajectory import run


def field_scan(N: int, beta: float, fields, sweeps: int, *, seed: int = 0, backend: str = 'numba') -> list[dict]:
    rows = []
    for lam in fields:
        for dynamics in ('metropolis','heatbath','sw-ghost','sw-size'):
            for init in ('minus','plus'):
                state, series = run(IsingModel(N,beta,lam), sweeps, init=init, seed=seed, dynamics=dynamics, backend=backend)
                tail = [r['magnetization'] for r in series if r['sweep'] >= sweeps/2]
                rows.append(dict(lam=lam,dynamics=dynamics,init=init,mean_m=float(np.mean(tail)),largest_plus_fraction=largest_plus_fraction(state)))
    return rows
