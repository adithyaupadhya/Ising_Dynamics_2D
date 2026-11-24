"""Transition kernels, independent of experiments and rendering."""
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def _components(n, a, b):
    """Label connected components of the activated bond graph."""
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]

def _sw_step_scipy(s, beta, h, rng, ghost):
    """Sparse-graph reference for either cluster transition rule."""
    N = s.shape[0] - 2
    n = N * N
    MINUS, PLUS = n, n + 1
    p = - np.expm1(-2.0 * beta)
    p_ghost = - np.expm1(-2.0 * beta * h)
    core = s[1:-1, 1:-1]
    ids = np.arange(n).reshape(N, N)
    edge_u, edge_v = [], []
    hor = (core[:, :-1] == core[:, 1:]) & (rng.random((N, N - 1)) < p)
    ver = (core[:-1, :] == core[1:, :]) & (rng.random((N - 1, N)) < p)
    edge_u += [ids[:, :-1][hor], ids[:-1, :][ver]]
    edge_v += [ids[:, 1:][hor], ids[1:, :][ver]]
    neg = core == -1
    for flag, site in ((neg[:, 0], ids[:, 0]), (neg[:, -1], ids[:, -1]),
                       (neg[0, :], ids[0, :]), (neg[-1, :], ids[-1, :])):
        m = flag & (rng.random(N) < p)
        edge_u.append(site[m])
        edge_v.append(np.full(int(m.sum()), MINUS))
    if ghost:
        g = (core == 1) & (rng.random((N, N)) < p_ghost)
        edge_u.append(ids[g])
        edge_v.append(np.full(int(g.sum()), PLUS))
    lab = _components(n + 1 + int(ghost), np.concatenate(edge_u), np.concatenate(edge_v))
    count = int(lab.max()) + 1
    if ghost:
        new = rng.integers(0, 2, count).astype(np.int8) * 2 - 1
        new[lab[PLUS]] = 1
    else:
        sizes = np.bincount(lab[:n], minlength=count)
        prob = 1 / (1 + np.exp(-2 * beta * h * sizes))
        new = np.where(rng.random(count) < prob, 1, -1).astype(np.int8)
    new[lab[MINUS]] = -1
    s[1:-1, 1:-1] = new[lab[:n]].reshape(N, N)


@njit(cache=True)
def _root(parent, vertex):
    """Find a component representative with path halving."""
    while parent[vertex] != vertex:
        parent[vertex] = parent[parent[vertex]]
        vertex = parent[vertex]
    return vertex


@njit(cache=True)
def _join(parent, sizes, a, b):
    """Merge components by size."""
    a, b = _root(parent, a), _root(parent, b)
    if a != b:
        if sizes[a] < sizes[b]:
            a, b = b, a
        parent[b] = a
        sizes[a] += sizes[b]


@njit(cache=True)
def _sw_step_numba(s, beta, h, rng, ghost):
    """Sample bonds and merge directly, then recolor each free component."""
    N = s.shape[0] - 2
    n = N * N
    minus, plus = n, n + 1
    p = -np.expm1(-2 * beta)
    p_ghost = -np.expm1(-2 * beta * h) if ghost else 0.0
    vertices = n + 2 if ghost else n + 1
    parent = np.arange(vertices)
    sizes = np.ones(vertices, np.int64)
    for i in range(N):
        for j in range(N):
            a = i * N + j
            spin = s[i + 1, j + 1]
            # Each interior edge is tested exactly once.
            if j + 1 < N and spin == s[i + 1, j + 2] and rng.random() < p:
                _join(parent, sizes, a, a + 1)
            if i + 1 < N and spin == s[i + 2, j + 1] and rng.random() < p:
                _join(parent, sizes, a, a + N)
            if spin == -1:
                # Corners have two boundary edges (N=1 has four).
                if i == 0 and rng.random() < p:
                    _join(parent, sizes, a, minus)
                if i == N - 1 and rng.random() < p:
                    _join(parent, sizes, a, minus)
                if j == 0 and rng.random() < p:
                    _join(parent, sizes, a, minus)
                if j == N - 1 and rng.random() < p:
                    _join(parent, sizes, a, minus)
            elif ghost and rng.random() < p_ghost:
                _join(parent, sizes, a, plus)

    colors = np.zeros(vertices, np.int8)
    colors[_root(parent, minus)] = -1
    if ghost:
        colors[_root(parent, plus)] = 1
    for a in range(n):
        root = _root(parent, a)
        if colors[root] == 0:
            # Free components contain no auxiliary vertices.
            prob = .5 if ghost else 1 / (1 + np.exp(-2 * beta * h * sizes[root]))
            colors[root] = 1 if rng.random() < prob else -1
        s[a // N + 1, a % N + 1] = colors[root]


def sw_ghost_python(s, beta, h, rng):
    _sw_step_scipy(s, beta, h, rng, True)


def sw_size_python(s, beta, h, rng):
    _sw_step_scipy(s, beta, h, rng, False)


def sw_ghost_numba(s, beta, h, rng):
    _sw_step_numba(s, beta, h, rng, True)


def sw_size_numba(s, beta, h, rng):
    _sw_step_numba(s, beta, h, rng, False)


CLUSTER_KERNELS = {
    ('sw-ghost', 'python'): sw_ghost_python,
    ('sw-ghost', 'numba'): sw_ghost_numba,
    ('sw-size', 'python'): sw_size_python,
    ('sw-size', 'numba'): sw_size_numba,
}


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
