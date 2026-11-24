"""Right-censored survival estimates, including tied event/censor times."""
import numpy as np


def kaplan_meier(times, events) -> list[dict]:
    """Return risk counts, survival and Greenwood standard errors at observed times."""
    times, events = np.asarray(times,float), np.asarray(events)
    if times.ndim != 1 or not len(times) or times.shape != events.shape or not np.isfinite(times).all() or (times<0).any() or not np.isin(events,[0,1]).all():
        raise ValueError('Require nonnegative finite times and binary event indicators')
    survival, greenwood, at_risk = 1., 0., len(times)
    rows = []
    for t in np.unique(times):
        mask = times == t
        deaths = int(events[mask].sum())
        censored = int(mask.sum())-deaths
        survival *= 1-deaths/at_risk
        if deaths < at_risk:
            greenwood += deaths/(at_risk*(at_risk-deaths))
        rows.append(dict(time=float(t),at_risk=at_risk,events=deaths,censored=censored,survival=survival,se=float(survival*np.sqrt(greenwood))))
        at_risk -= deaths+censored
    return rows


def median_survival(curve) -> float | None:
    """Return first time survival <= 1/2, or None if not identified."""
    return next((row['time'] for row in curve if row['survival'] <= .5),None)


def restricted_mean(curve, horizon: float) -> float:
    """Integrate survival up to an observed horizon; no tail extrapolation."""
    if not curve or not np.isfinite(horizon) or not 0 <= horizon <= curve[-1]['time']:
        raise ValueError('Horizon must lie within follow-up')
    previous, survival, total = 0., 1., 0.
    for row in curve:
        t = min(row['time'],horizon)
        total += survival*(t-previous)
        if t == horizon:
            break
        previous,survival = t,row['survival']
    return total
