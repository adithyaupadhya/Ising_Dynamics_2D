"""Boundary profiles; row 1 is top, column 1 is left."""
import numpy as np


def boundary_magnetization(state):
    core = state[1:-1, 1:-1]
    n = len(core)
    x = np.arange(n)
    d = np.minimum(x, n-1-x)
    layers = np.minimum(d[:,None], d[None,:]).ravel()
    return (np.bincount(layers, weights=core.ravel())/np.bincount(layers)).tolist()


def interface_heights(mask, trim=0):
    """Distance from fixed frame to first droplet site; None for missing rays."""
    n = len(mask)
    if not isinstance(trim, int) or trim<0 or 2*trim>=n:
        raise ValueError('trim must leave at least one ray')
    result = {}
    for side, rays in [('top',mask.T), ('bottom',mask[::-1].T),
                       ('left',mask), ('right',mask[:,::-1])]:
        rays = rays[trim:n-trim]
        values = [int(np.argmax(ray))+1 if ray.any() else None for ray in rays]
        valid = np.array([v for v in values if v is not None], dtype=float)
        result[side] = dict(heights=values, coverage=len(valid)/len(values),
                            mean=float(valid.mean()) if len(valid) else None,
                            variance=float(valid.var()) if len(valid) else None,
                            minimum=int(valid.min()) if len(valid) else None,
                            maximum=int(valid.max()) if len(valid) else None)
    return result
