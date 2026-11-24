"""Time identical random-site kernels, including random generation, after warmup."""
import time
import numpy as np
from ising import IsingModel, Simulator


def elapsed(sim, repeats):
    sim.advance(2)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        sim.advance(repeats)
        timings.append((time.perf_counter()-start)/repeats)
    return float(np.median(timings))


def run_benchmark(sizes) -> list[dict]:
    rows = []
    for n in sizes:
        for dynamics in ('heatbath','metropolis'):
            times = [elapsed(Simulator(IsingModel(n,.5,3.), 'random',42,dynamics=dynamics,backend=b),
                             max(5,32768//(n*n))) for b in ('python','numba')]
            rows.append(dict(N=n,dynamics=dynamics,python_ms=times[0]*1000,numba_ms=times[1]*1000,speedup=times[0]/times[1]))
    return rows
