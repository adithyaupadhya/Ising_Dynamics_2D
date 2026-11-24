"""Monte Carlo equilibrium diagnostics against exact enumeration."""
from ising import IsingModel, Simulator
from ising.exact import exact_small
from ising.observables import bond_sum
from ising.analysis.statistics import batch_mean


def equilibrium_check(model: IsingModel, steps: int = 100000, burn: int = 500, *, seed: int = 1, backend: str = 'numba') -> tuple[dict,list[dict]]:
    if steps < 20 or burn < 0:
        raise ValueError('Require at least 20 samples and nonnegative burn-in')
    w,bonds,msum = exact_small(model.N,model.beta,model.lam)
    exact = {'m':float(w@msum/model.num_sites),'bond sum':float(w@bonds),'P(all -)':float(w[0])}
    rows = []
    for dynamics in ('heatbath','metropolis','sw-ghost','sw-size'):
        sim = Simulator(model,seed=seed,dynamics=dynamics,backend=backend)
        sim.advance(burn)
        series = {k:[] for k in exact}
        for _ in range(steps):
            sim.advance()
            series['m'].append(sim.magnetization)
            series['bond sum'].append(bond_sum(sim.state))
            series['P(all -)'].append(float(sim.magnetization == -1))
        for observable,expected in exact.items():
            mean,se = batch_mean(series[observable])
            rows.append(dict(dynamics=dynamics,observable=observable,mean=mean,se=se,z=(mean-expected)/se if se else None))
    return exact, rows
