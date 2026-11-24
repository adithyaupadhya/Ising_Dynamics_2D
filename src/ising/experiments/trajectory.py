"""Record single trajectories without console or file I/O."""
from ising import IsingModel, Simulator


def run(model: IsingModel, sweeps: int, *, init: str = 'minus', seed: int = 0,
        record_every: int = 10, dynamics: str = 'heatbath', backend: str = 'numba',
        observables: dict | None = None) -> tuple:
    """Return a final state snapshot and observations at zero, intervals, and final time."""
    if not isinstance(sweeps,int) or not isinstance(record_every,int) or sweeps < 0 or record_every < 1:
        raise ValueError('Invalid sweep count or recording interval')
    sim = Simulator(model, init, seed, dynamics=dynamics, backend=backend)
    options = {'clusters': False} if observables is None else observables
    series = [sim.observe(**options)]
    while sim.t < sweeps:
        sim.advance(min(record_every, int(sweeps-sim.t)))
        series.append(sim.observe(**options))
    return sim.state.copy(), series
