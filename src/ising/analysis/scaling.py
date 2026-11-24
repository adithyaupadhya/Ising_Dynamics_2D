"""Descriptive finite-size fits; not proof of asymptotic scaling."""
import numpy as np
from scipy.stats import linregress
from .survival import kaplan_meier, median_survival


def exponential_fit(sizes, times) -> dict:
    """Fit log(time) = rate*N + intercept to positive characteristic times."""
    x,y = np.asarray(sizes,float),np.asarray(times,float)
    if x.ndim != 1 or x.shape != y.shape or len(x)<3 or len(np.unique(x))<3 or not np.isfinite(x).all() or not np.isfinite(y).all() or (x<=0).any() or (y<=0).any():
        raise ValueError('Need at least three distinct sizes and positive finite times')
    fit = linregress(x,np.log(y))
    return dict(rate=float(fit.slope),intercept=float(fit.intercept),r_squared=float(fit.rvalue**2),regression_se=float(fit.stderr))


def passage_medians(rows) -> list[dict]:
    """Group one fixed-parameter experiment by size, respecting censoring."""
    groups = {}
    reference = None
    keys = ('beta','lambda','dynamics','backend','init','threshold')
    for row in rows:
        signature = tuple(row.get(k) for k in keys)
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError('Analyze one fixed-parameter experiment at a time')
        groups.setdefault(int(row['N']),[]).append(row)
    return [dict(N=n,median=median_survival(kaplan_meier([float(r['sweeps']) for r in group],[int(r['event']) for r in group]))) for n,group in sorted(groups.items())]
