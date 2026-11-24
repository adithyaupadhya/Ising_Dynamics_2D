# Implementation and performance

[Home](../README.md) · [Dynamics](dynamics.md) · [Observables](observables.md)

## Reference and compiled kernels

`src/ising/kernels/local.py` defines `random_updates_python` and its compiled counterpart, `random_updates_numba = njit(cache=True)(random_updates_python)`. Both heat bath and Metropolis use this same sequential loop. The Python reference already uses probability tables and shared random inputs, so benchmarks isolate the benefit of compilation rather than compare against an artificially slow algorithm. Numba remains an installed dependency even when selecting Python. Both Swendsen–Wang variants use a sparse SciPy graph for the `python` backend and a compiled disjoint-set (union–find) kernel for `numba`. Each activated bond merges two components directly; path compression and union by size avoid constructing a sparse matrix. The size-based variant evaluates its logistic probability once per free component, without a lookup table. The two cluster backends consume random draws in different orders: reproducibility holds within each backend, not across backends.

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
| 64 | heatbath | 3.6516 | 0.0952 | 38.37× |
| 64 | metropolis | 3.8254 | 0.1095 | 34.95× |
| 64 | sw-ghost | 0.5607 | 0.1498 | 3.74× |
| 64 | sw-size | 0.4789 | 0.1478 | 3.24× |
| 256 | heatbath | 61.4533 | 1.9057 | 32.25× |
| 256 | metropolis | 60.1143 | 1.4124 | 42.56× |
| 256 | sw-ghost | 6.4319 | 2.2538 | 2.85× |
| 256 | sw-size | 6.1703 | 2.0274 | 3.04× |

These measurements were obtained on an AMD EPYC 9V74 CPU, Linux x86-64, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0 and Numba 0.65.1. The full results are in [benchmark_results.csv](benchmark_results.csv). These are simulator sweep timings, including thermodynamic bookkeeping but excluding rendering and other observables. Cluster backends implement the same transition law for each variant, not identical seeded trajectories. Speedup measures execution cost, not relaxation.

To reproduce the current benchmark on your machine:

```bash
ising benchmark --size 32 64 128 256
```

Record hardware, OS, Python and dependency versions with any newly reported timings.

## Geometry boundary

The supported model is still the two-dimensional square box with a fixed minus
frame. `model.geometry` returns an immutable `SquareLattice(N)`; no new lattice
or graph is implemented. The constructor `IsingModel(N, beta, lam)` and the
field scaling `h = lam/N` are unchanged.

`ising.geometry.square` owns the state layout, dynamic vertex count, physical
bond count, coordination number, critical inverse temperature, padded-index
mapping, physical edge views, boundary validation and NW geometric connectivity.
The geometry also selects the square-specific update kernels and observable
implementation. `Simulator` and `MonotoneCoupling` use the geometry to construct
states, choose kernels once, and convert update counts to sweeps. The random
stream uses the geometry's site mapping but retains the original draw order.
Energy bookkeeping now uses the number of sites and bonds rather than formulas
for a square box; `thermodynamics(N, ...)` remains the square-box convenience API.

For future extensions:

- A hypercubic lattice can supply a different padded shape, site mapping, bond
  views, degree and specialized kernels. The same state/clock/stream structure
  can be reused. A triangular or honeycomb lattice may instead use skew or
  parity-dependent neighbors and its own boundary rules.
- The local kernel arguments and cluster kernel are selected outside the update
  loops. Existing square kernels intentionally retain their explicit neighbor
  accesses and bond sampling order. Generic or new optimized kernels can be
  introduced alongside them, rather than replacing tested code now.
- Geometric observables are still square-specific: the NW matrix is only a
  geometric convention, not an interaction graph. Other planar lattices require
  their own contour topology and physical embedding; higher-dimensional lattices
  require surface observables. The current observable names are retained.
- Exact enumeration uses the physical bond views but the public `exact_small`
  function, CLI, experiment parameter schemas and visualization remain square
  specific. Generalizing the model constructor and field scaling is a separate
  future API decision, not part of this refactor.

No per-site Python geometry dispatch or explicit adjacency matrix is introduced.
