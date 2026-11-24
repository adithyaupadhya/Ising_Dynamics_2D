"""Transition kernels, independent of experiments and rendering."""
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def _components(num_vertices, edge_u, edge_v):
    """Label connected components of the activated bond graph."""
    graph = coo_matrix((np.ones(len(edge_u), dtype=np.int8), (edge_u, edge_v)), shape=(num_vertices, num_vertices))
    return connected_components(graph, directed=False)[1]

def _sw_step_scipy(spins, beta, h, rng, ghost):
    """Sparse-graph reference for either cluster transition rule."""
    N = spins.shape[0] - 2
    num_sites = N * N
    MINUS, PLUS = num_sites, num_sites + 1
    bond_prob = - np.expm1(-2.0 * beta)
    ghost_bond_prob = - np.expm1(-2.0 * beta * h)
    interior = spins[1:-1, 1:-1]
    site_ids = np.arange(num_sites).reshape(N, N)
    edge_u, edge_v = [], []
    horizontal_bonds = (interior[:, :-1] == interior[:, 1:]) & (rng.random((N, N - 1)) < bond_prob)
    vertical_bonds = (interior[:-1, :] == interior[1:, :]) & (rng.random((N - 1, N)) < bond_prob)
    edge_u += [site_ids[:, :-1][horizontal_bonds], site_ids[:-1, :][vertical_bonds]]
    edge_v += [site_ids[:, 1:][horizontal_bonds], site_ids[1:, :][vertical_bonds]]
    minus_mask = interior == -1
    for boundary_minus_mask, boundary_site_ids in ((minus_mask[:, 0], site_ids[:, 0]), (minus_mask[:, -1], site_ids[:, -1]),
                       (minus_mask[0, :], site_ids[0, :]), (minus_mask[-1, :], site_ids[-1, :])):
        active_boundary_bonds = boundary_minus_mask & (rng.random(N) < bond_prob)
        edge_u.append(boundary_site_ids[active_boundary_bonds])
        edge_v.append(np.full(int(active_boundary_bonds.sum()), MINUS))
    if ghost:
        active_ghost_bonds = (interior == 1) & (rng.random((N, N)) < ghost_bond_prob)
        edge_u.append(site_ids[active_ghost_bonds])
        edge_v.append(np.full(int(active_ghost_bonds.sum()), PLUS))
    component_labels = _components(num_sites + 1 + int(ghost), np.concatenate(edge_u), np.concatenate(edge_v))
    num_components = int(component_labels.max()) + 1
    if ghost:
        component_spins = rng.integers(0, 2, num_components).astype(np.int8) * 2 - 1
        component_spins[component_labels[PLUS]] = 1
    else:
        component_sizes = np.bincount(component_labels[:num_sites], minlength=num_components)
        plus_probabilities = 1 / (1 + np.exp(-2 * beta * h * component_sizes))
        component_spins = np.where(rng.random(num_components) < plus_probabilities, 1, -1).astype(np.int8)
    component_spins[component_labels[MINUS]] = -1
    spins[1:-1, 1:-1] = component_spins[component_labels[:num_sites]].reshape(N, N)


@njit(cache=True)
def _root(parent, vertex):
    """Find a component representative with path halving."""
    while parent[vertex] != vertex:
        parent[vertex] = parent[parent[vertex]]
        vertex = parent[vertex]
    return vertex


@njit(cache=True)
def _join(parent, component_sizes, u, v):
    """Merge components by size."""
    u, v = _root(parent, u), _root(parent, v)
    if u != v:
        if component_sizes[u] < component_sizes[v]:
            u, v = v, u
        parent[v] = u
        component_sizes[u] += component_sizes[v]


@njit(cache=True)
def _sw_step_numba(spins, beta, h, rng, ghost):
    """Sample bonds and merge directly, then recolor each free component."""
    N = spins.shape[0] - 2
    num_sites = N * N
    minus_vertex, plus_vertex = num_sites, num_sites + 1
    bond_prob = -np.expm1(-2 * beta)
    ghost_bond_prob = -np.expm1(-2 * beta * h) if ghost else 0.0
    num_vertices = num_sites + 2 if ghost else num_sites + 1
    parent = np.arange(num_vertices)
    component_sizes = np.ones(num_vertices, np.int64)
    for i in range(N):
        for j in range(N):
            site_id = i * N + j
            spin = spins[i + 1, j + 1]
            # Each interior edge is tested exactly once.
            if j + 1 < N and spin == spins[i + 1, j + 2] and rng.random() < bond_prob:
                _join(parent, component_sizes, site_id, site_id + 1)
            if i + 1 < N and spin == spins[i + 2, j + 1] and rng.random() < bond_prob:
                _join(parent, component_sizes, site_id, site_id + N)
            if spin == -1:
                # Corners have two boundary edges (N=1 has four).
                if i == 0 and rng.random() < bond_prob:
                    _join(parent, component_sizes, site_id, minus_vertex)
                if i == N - 1 and rng.random() < bond_prob:
                    _join(parent, component_sizes, site_id, minus_vertex)
                if j == 0 and rng.random() < bond_prob:
                    _join(parent, component_sizes, site_id, minus_vertex)
                if j == N - 1 and rng.random() < bond_prob:
                    _join(parent, component_sizes, site_id, minus_vertex)
            elif ghost and rng.random() < ghost_bond_prob:
                _join(parent, component_sizes, site_id, plus_vertex)

    component_spins = np.zeros(num_vertices, np.int8)
    component_spins[_root(parent, minus_vertex)] = -1
    if ghost:
        component_spins[_root(parent, plus_vertex)] = 1
    for site_id in range(num_sites):
        component_root = _root(parent, site_id)
        if component_spins[component_root] == 0:
            # Free components contain no auxiliary vertices.
            plus_prob = .5 if ghost else 1 / (1 + np.exp(-2 * beta * h * component_sizes[component_root]))
            component_spins[component_root] = 1 if rng.random() < plus_prob else -1
        spins[site_id // N + 1, site_id % N + 1] = component_spins[component_root]


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
    local_fields = 2 * np.arange(5) - 4 + model.h
    heatbath_probs = np.exp(-np.logaddexp(0., -2 * model.beta * local_fields))
    metropolis_probs = np.exp(-model.beta * np.maximum(2 * np.array([-1, 1])[:, None] * local_fields, 0))
    return heatbath_probs, metropolis_probs


def random_updates_python(flat, w, sites, uniforms, hb, mp, heatbath,
                          spin_sum, length, direction=0, threshold=0.0, area=1):
    """Sequential random-site reference loop; stop at first threshold crossing.

    direction=0 disables stopping; +1 means upward, -1 downward.
    Return number of attempts consumed and updated total spin and broken-bond count.
    """
    for update_idx in range(len(sites)):
        site = sites[update_idx]
        num_plus_neighbors = (int(flat[site-1]) + int(flat[site+1]) + int(flat[site-w])
             + int(flat[site+w]) + 4) // 2
        old_spin = int(flat[site])
        new_spin = old_spin
        if heatbath:
            new_spin = 1 if uniforms[update_idx] < hb[num_plus_neighbors] else -1
        elif uniforms[update_idx] < mp[(old_spin+1)//2, num_plus_neighbors]:
            new_spin = -old_spin
        flat[site] = new_spin
        spin_sum += new_spin - old_spin
        length += ((old_spin-new_spin)//2)*(2*num_plus_neighbors-4)
        if direction and direction * spin_sum / area >= direction * threshold:
            return update_idx + 1, spin_sum, length
    return len(sites), spin_sum, length


# Compile the very same reference function, not a different transition rule.
random_updates_numba = njit(cache=True)(random_updates_python)
