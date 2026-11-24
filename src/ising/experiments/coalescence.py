"""Replicated extremal-chain coalescence experiments."""
import time
from ising import IsingModel, MonotoneCoupling
from ising.model import validate_algorithm
from .runner import make_tasks, run_trials, TrialResults


def trial(task: dict) -> dict:
    model = task['model']
    start = time.perf_counter()
    pair = MonotoneCoupling(model, task['seed'], backend=task['backend']).advance(task['limit'])
    return dict(N=model.N, beta=model.beta, **{'lambda':model.lam}, backend=task['backend'],
                replicate=task['replicate'], seed=task['seed'], limit=task['limit'], **pair.observe(),
                event=int(pair.coalesced), sweeps=pair.t, elapsed_seconds=time.perf_counter()-start)


def coalescence(sizes, *, beta: float = .5, lam: float = 3., backend: str = 'numba',
                limit: int = 10000, replicates: int = 20, seed: int = 42, workers: int = 1,
                output=None, resume: bool = False, progress=None) -> TrialResults:
    """Return independent coalescence trials, with optional parallelism and checkpointing."""
    validate_algorithm('heatbath', backend)
    if not isinstance(limit,int) or limit < 0:
        raise ValueError('Invalid coalescence limit')
    sizes = list(sizes)
    models = [IsingModel(n,beta,lam) for n in sizes]
    options = dict(backend=backend, limit=limit)
    parameters = dict(experiment='coalescence', sizes=sizes,beta=beta,lam=lam,replicates=replicates,seed=seed,**options)
    return run_trials(trial, make_tasks(models,replicates,seed,options),parameters,
                      workers=workers,output=output,resume=resume,progress=progress)
