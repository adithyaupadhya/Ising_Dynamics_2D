import itertools
import numpy as np
from ising.observables import (plus_components, droplet_mask, peierls_contours,
                               interface_length, droplet_geometry)


def padded(interior):
    return np.pad(np.asarray(interior), 1, constant_values=-1)


def test_diagonal_components():
    for interior, sizes in [([[-1, 1], [1, -1]], [2]),
                            ([[1, -1], [-1, 1]], [1, 1])]:
        labels, counts = plus_components(padded(interior))
        assert counts.tolist() == sizes
        assert droplet_mask(labels, counts).sum() == sizes[0]


def test_escape_channels():
    for allowed in (True, False):
        interior = np.ones((7, 7), dtype=int)
        for i in range(4):
            interior[i, 6-i if allowed else i] = -1
        mask = droplet_mask(*plus_components(padded(interior)))
        assert mask.sum() == (45 if allowed else 48)
        # Close the channel at the boundary: all remaining holes fill.
        interior[0, 6 if allowed else 0] = 1
        assert droplet_mask(*plus_components(padded(interior))).all()


def edges(contour):
    return {(tuple(a), tuple(b)) for a, b in zip(contour[:-1], contour[1:])}


def check_configuration(state):
    labels, sizes = plus_components(state)
    assert sizes.sum() == np.count_nonzero(state == 1)
    assert np.array_equal(np.bincount(labels.ravel())[1:], sizes)
    contours = peierls_contours(state)
    all_edges = set()
    for contour in contours:
        assert np.array_equal(contour[0], contour[-1])
        assert np.all(np.abs(np.diff(contour, axis=0)).sum(axis=1) == 1)
        current = edges(contour)
        assert len(current) == len(contour)-1
        assert not all_edges.intersection(current)
        assert not all_edges.intersection((b, a) for a, b in current)
        all_edges.update(current)
    assert len(all_edges) == interface_length(state)
    mask = droplet_mask(labels, sizes)
    if mask.any():
        outer = peierls_contours(padded(np.where(mask, 1, -1)))
        assert len(outer) == 1
        assert any(edges(outer[0]) == edges(c) for c in contours)
        assert len(outer[0])-1 == droplet_geometry(mask)['droplet_perimeter']


def test_exhaustive():
    for values in itertools.product((-1, 1), repeat=9):
        check_configuration(padded(np.array(values).reshape(3, 3)))


def test_larger_random_and_nested():
    rng = np.random.default_rng(42)
    for p in (.2, .5, .8):
        for _ in range(20):
            check_configuration(padded(np.where(rng.random((16, 16)) < p, 1, -1)))
    ring = np.ones((9, 9), dtype=int)
    ring[1:-1, 1:-1] = -1
    ring[3:6, 3:6] = 1
    check_configuration(padded(ring))
