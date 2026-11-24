"""Padded lattice initialization."""
import numpy as np

def make_state(N, init, rng):
    s = -np.ones((N + 2, N + 2), dtype=np.int8)
    if init == "plus":
        s[1:-1, 1:-1] = 1
    elif init == "random":
        s[1:-1, 1:-1] = rng.choice([-1, 1], size=(N, N)).astype(np.int8)
    elif init != "minus":
        raise ValueError(init)
    return s
