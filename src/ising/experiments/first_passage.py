"""Replicated elementary-update first-passage experiments."""
import time
from ising import IsingModel, Simulator
from ising.model import validate_algorithm
from .runner import make_tasks, run_trials, TrialResults


def trial(task: dict) -> dict:
    model = task['model']
    init, threshold, limit = (task[k] for k in ('init', 'threshold', 'limit'))
    dynamics, backend = task['dynamics'], task['backend']
    sim = Simulator(model, init, task['seed'], dynamics=dynamics, backend=backend)
    start = time.perf_counter()
    event = sim.first_passage(limit, threshold, 1 if init == 'minus' else -1)
    return dict(N=model.N, beta=model.beta, **{'lambda': model.lam}, h=model.h,
                dynamics=dynamics, backend=backend,
                init=init, threshold=threshold, replicate=task['replicate'], seed=task['seed'],
                limit_sweeps=limit, sweeps=sim.t, updates=sim.updates,
                event=int(event), final_m=sim.magnetization, elapsed_seconds=time.perf_counter()-start)


def first_passage(sizes, *, beta: float = .5, lam: float = 3., dynamics: str = 'heatbath',
                  backend: str = 'numba', init: str = 'minus', threshold: float = 0.,
                  limit: int = 10000, replicates: int = 20, seed: int = 42, workers: int = 1,
                  output=None, resume: bool = False, progress=None) -> TrialResults:
    """Return independent passage trials; optionally checkpoint to a study directory."""
    validate_algorithm(dynamics, backend)
    if init not in ('minus','plus') or not -1 < threshold < 1 or not isinstance(limit,int) or limit < 0:
        raise ValueError('Invalid first-passage parameters')
    sizes = list(sizes)
    models = [IsingModel(n, beta, lam) for n in sizes]
    options = dict(dynamics=dynamics, backend=backend, init=init, threshold=threshold, limit=limit)
    parameters = dict(experiment='passage', sizes=sizes, beta=beta, lam=lam, replicates=replicates, seed=seed, **options)
    return run_trials(trial, make_tasks(models, replicates, seed, options), parameters,
                      workers=workers, output=output, resume=resume, progress=progress)
