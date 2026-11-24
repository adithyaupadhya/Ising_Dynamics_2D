"""Uncertainty estimates for independent or sufficiently stationary samples."""
import numpy as np


def batch_mean(x, nb: int = 20) -> tuple[float,float]:
    """Equal-length contiguous batch means; discard incomplete tail consistently."""
    x = np.asarray(x,float)
    if x.ndim != 1 or len(x) < nb or nb < 2 or not np.isfinite(x).all():
        raise ValueError('Need finite observations and at least two nonempty batches')
    k = len(x)//nb
    means = x[:k*nb].reshape(nb,k).mean(axis=1)
    return float(means.mean()), float(means.std(ddof=1)/np.sqrt(nb))


def bootstrap_mean(x, *, replicates: int = 2000, confidence: float = .95, seed: int = 0) -> tuple[float,float]:
    """Percentile interval for the mean of independent observations, not correlated MCMC samples."""
    x = np.asarray(x,float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or replicates < 1 or not 0 < confidence < 1:
        raise ValueError('Invalid bootstrap parameters')
    rng = np.random.default_rng(seed)
    means = [rng.choice(x,size=len(x),replace=True).mean() for _ in range(replicates)]
    return tuple(float(v) for v in np.quantile(means, [(1-confidence)/2,(1+confidence)/2]))
