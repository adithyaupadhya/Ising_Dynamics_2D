# Implementation and performance

[Home](../README.md) · [Dynamics](dynamics.md) · [Observables](observables.md)

## Reference and compiled kernels

`src/ising/kernels.py` defines `random_updates_python` and its compiled counterpart, `random_updates_numba = njit(cache=True)(random_updates_python)`. Both heat bath and Metropolis use this same sequential loop. The Python reference already uses probability tables and shared random inputs, so benchmarks isolate the benefit of compilation rather than compare against an artificially slow algorithm. Numba remains an installed dependency even when selecting Python. Swendsen–Wang uses the same SciPy implementation for either backend.

## Probability tables and padded arrays

Each interior site has four neighbors, including fixed boundary spins when needed. The padded `int8` array makes these accesses uniform, without special boundary branches. The neighbor sum takes only five values, $0, \pm 2, \pm 4$. Heat bath therefore needs five precomputed probabilities and Metropolis ten (five sums for each spin sign). No repeated exponentiation is needed in the update loop.

## Incremental observables

Let $M$ be the interior spin sum and $L$ the number of disagreeing physical interaction edges. If an update changes spin $s$ to $s'$ with neighbor sum $k$, then

$$\Delta M=s'-s,\qquad \Delta L=\frac{s-s'}{2}k,$$

$$\Delta H=-(s'-s)(k+h_N).$$

The local loop updates these counts without rescanning the lattice. Cluster steps and state replacement recompute both counts. See [Observables](observables.md) for the energy convention.

## Random streams and reproducibility

`random_stream.py` generates blocks of $N^2$ uniform site indices in $[0,N^2)$, maps them to padded indices, and generates independent uniform variates for decisions. `peek` exposes pending draws and `consume` removes only attempts actually executed. Unused draws remain buffered if a call stops mid-block. Consequently splitting a seeded trajectory into smaller calls does not change its evolution.

The simulator and coupling share this implementation. The two local backends consume identical arrays in identical order. Th updates are performed sequentially. The browser demo has a separate JavaScript implementation, not an identical Python random stream.

```python
import numpy as np
from ising import IsingModel, Simulator

model = IsingModel(16, .5, 3)
a = Simulator(model, backend="python", seed=42)
b = Simulator(model, backend="numba", seed=42)
a.advance(10)
b.advance(4)
b.advance(6)
assert np.array_equal(a.state, b.state)
```

Model and algorithm properties are read-only to avoid stale tables. `sim.state` is a read-only padded view; use `sim.state.copy()` for a snapshot and `sim.set_state(array)` for validated replacement without resetting time or the random stream. `advance(k)` performs $k$ sweep updates, while `advance_updates(k)` performs $k$ elementary updates; `updates` counts elementary updates and `t` is the corresponding sweep-equivalent time.

## Benchmarks

The benchmark warms each backend for two sweeps, then reports the median of three timed batches, including random generation and incremental bookkeeping. Each batch uses `max(5, 32768 // (N*N))` sweeps. The current benchmark uses $\beta=0.5$, $\lambda=3$, random initialization and seed 42. First-use compilation is excluded by warmup; cached code may reduce startup cost.

| N | Dynamics | Python (ms/sweep) | Numba (ms/sweep) | Speedup |
|---:|---|---:|---:|---:|
| 32 | Heat bath | 1.27150 | 0.03975 | 31.99× |
| 64 | Heat bath | 4.82762 | 0.11795 | 40.93× |
| 128 | Heat bath | 18.41081 | 0.46571 | 39.53× |
| 256 | Heat bath | 72.41165 | 1.84637 | 39.22× |

These are existing example measurements from [benchmark_results.csv](benchmark_results.csv), which also contains Metropolis results. These are not  performance guarantees. Both local algorithms require $O(N^2)$ work per sweep. The speedup concerns implementation overhead, not stochastic relaxation.

To reproduce the current benchmark on your machine:

```bash
ising benchmark --size 32 64 128 256
```

Record hardware, OS, Python and dependency versions with any newly reported timings.
