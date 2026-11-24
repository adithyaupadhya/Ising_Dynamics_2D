"""Transition kernels, independent of experiments and rendering."""
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def _components(n, a, b):
    """Label connected components of the activated bond graph."""
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]

def sw_step(s, beta, h, rng):
    """One Swendsen-Wang step with a ghost + spin (field) and a fixed - frame."""
    N = s.shape[0] - 2
    n = N * N
    MINUS, PLUS = n, n + 1
    p = 1.0 - np.exp(-2.0 * beta)
    pg = 1.0 - np.exp(-2.0 * beta * h)
    core = s[1:-1, 1:-1]
    ids = np.arange(n).reshape(N, N)
    ea, eb = [], []
    hor = (core[:, :-1] == core[:, 1:]) & (rng.random((N, N - 1)) < p)
    ver = (core[:-1, :] == core[1:, :]) & (rng.random((N - 1, N)) < p)
    ea += [ids[:, :-1][hor], ids[:-1, :][ver]]
    eb += [ids[:, 1:][hor], ids[1:, :][ver]]
    neg = core == -1
    for flag, site in ((neg[:, 0], ids[:, 0]), (neg[:, -1], ids[:, -1]),
                       (neg[0, :], ids[0, :]), (neg[-1, :], ids[-1, :])):
        m = flag & (rng.random(N) < p)
        ea.append(site[m])
        eb.append(np.full(int(m.sum()), MINUS))
    g = (core == 1) & (rng.random((N, N)) < pg)
    ea.append(ids[g])
    eb.append(np.full(int(g.sum()), PLUS))
    lab = _components(n + 2, np.concatenate(ea), np.concatenate(eb))
    new = rng.integers(0, 2, int(lab.max()) + 1).astype(np.int8) * 2 - 1
    new[lab[MINUS]] = -1
    new[lab[PLUS]] = 1
    s[1:-1, 1:-1] = new[lab[:n]].reshape(N, N)


def local_tables(model):
    """Build heat-bath and Metropolis lookup tables."""
    loc = 2 * np.arange(5) - 4 + model.h
    hb = np.exp(-np.logaddexp(0., -2 * model.beta * loc))
    mp = np.exp(-model.beta * np.maximum(2 * np.array([-1, 1])[:, None] * loc, 0))
    return hb, mp


def random_updates_python(flat, w, sites, uniforms, hb, mp, heatbath,
                          spin_sum, length, direction=0, threshold=0.0, area=1):
    """Sequential random-site reference loop; stop at first threshold crossing.

    direction=0 disables stopping; +1 means upward, -1 downward.
    Return number of attempts consumed and updated total spin and broken-bond count.
    """
    for i in range(len(sites)):
        p = sites[i]
        k = (int(flat[p-1]) + int(flat[p+1]) + int(flat[p-w])
             + int(flat[p+w]) + 4) // 2
        old = int(flat[p])
        new = old
        if heatbath:
            new = 1 if uniforms[i] < hb[k] else -1
        elif uniforms[i] < mp[(old+1)//2, k]:
            new = -old
        flat[p] = new
        spin_sum += new - old
        length += ((old-new)//2)*(2*k-4)
        if direction and direction * spin_sum / area >= direction * threshold:
            return i + 1, spin_sum, length
    return len(sites), spin_sum, length


# Compile the very same reference function, not a different transition rule.
random_updates_numba = njit(cache=True)(random_updates_python)
