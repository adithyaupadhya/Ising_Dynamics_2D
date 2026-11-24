# Dynamics and monotone coupling

[Home](../README.md) · [Model](model.md) · [Implementation](implementation.md)

All three choices of reversible Markov chain dynamics have as the equilibrium measure the Gibbs measure defined in [Model](model.md). Write $k_x=\sum_{y\sim x}\sigma_y$ and $F_x=k_x+h_N$.

## Heat bath

An interior site $x$ is chosen uniformly at random, independently at each step. Its spin is resampled as $+1$ with probability

$$p_x(\sigma)=\frac{1}{1+e^{-2\beta F_x}},$$

and as $-1$ otherwise. The spin can remain unchanged. This is a reversible single-site Gibbs update.

## Metropolis

Again, an interior site $x$ is chosen uniformly at random, independently at each step. If its current spin is $s$, the energy change for flipping it is $\Delta H=2sF_x$. The proposed flip is accepted with probability

$$\min\{1,e^{-\beta\Delta H}\}.$$

A rejected proposal leaves the configuration unchanged. This is also a reversible single-site Gibbs update.

## Swendsen–Wang

Random bonds are sampled conditional on the current configuration as follows:

- For each pair of neighbouring vertices, if they have the same spin, bonds between them are drawn independently with probability $1-e^{-2\beta}$. This includes when both vertices are in the interior, or one vertex is in the interior and one in the exterior. When both vertices are in the exterior, the bond is always drawn, so the boundary is wired.
- For all vertices with plus spins, a bond is drawn to a fixed plus ghost vertex with probability $1-e^{-2\beta h_N}$.
- Each vertex in a connected component is given the same sign. If the component contains the ghost vertex, it is assigned plus. If it contains the exterior vertices, it is assigned minus. All other components are independently assigned plus or minus uniformly at random. 

This is a cluster Gibbs sampler using the random cluster representation. Its bond components are not the same as the geometric spin clusters measured by the observables.

## Time conventions

| Property | Heat bath | Metropolis | Swendsen–Wang |
|---|---|---|---|
| Update | Site redraw | Flip attempt | Cluster step |
| Sweep | $N^2$ updates | $N^2$ updates | $1$ update |
| Backend | Python / Numba | Python / Numba | Shared SciPy implementation |
| Implemented monotone coupling | Yes | No | No |

 Equal sweep counts do not imply equal computational cost or comparable physical relaxation times. 

## Monotone coupling

The lower chain is started at all minus and the upper chain at all plus, and they both use the same random site and random uniform variate at every heat-bath update. The site resampling probability is increasing in the neighbor sum, so pointwise order is preserved. Each marginal follows the correct heat-bath dynamics.

```python
from ising import IsingModel, MonotoneCoupling

pair = MonotoneCoupling(IsingModel(32, .5, 3), seed=42, backend="numba")
pair.advance(100)
print(pair.observe())
```

`lower` and `upper` expose read-only padded arrays. Disagreements are maintained incrementally; `observe()` scans the configurations for their magnetizations. Both backends share the coupling kernel. `advance(sweeps)` and `advance_updates(count)` stop at the exact first coalescence, including within a sweep. Subsequent calls do nothing. `updates` counts attempts and `t` is attempts divided by $N^2$.

For the grand monotone coupling, with coalesence time $T_{\rm co}$ referring to the first (random) time that the upper and lower chain are equal,

$$\max_\sigma\|P_t(\sigma,\cdot)-\pi\|_{\rm TV}\leq\mathbb P(T_{\rm co}>t).$$

Here $t$ refers to the discrete update time (or its sweep-equivalent rescaling). An empirical survival estimate of the right-hand side requires statistical uncertainty; it is not itself a rigorous deterministic bound. See [Experiments](experiments.md) and [Analysis](analysis.md).

## Further reading

- R. H. Swendsen and J.-S. Wang, *Nonuniversal critical dynamics in Monte Carlo simulations* (1987).
- D. A. Levin and Y. Peres, *Markov Chains and Mixing Times*, second edition (2017).
