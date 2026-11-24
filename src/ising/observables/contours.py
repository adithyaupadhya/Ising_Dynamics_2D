"""Northwest-resolved Peierls contours and filled droplet geometry."""
import numpy as np
from scipy.ndimage import binary_fill_holes
from ..geometry.square import NW_CONNECTIVITY


def droplet_mask(labels, counts):
    if not len(counts):
        return np.zeros(labels.shape, dtype=bool)
    return binary_fill_holes(labels == (int(np.argmax(counts))+1),
                             structure=NW_CONNECTIVITY)


def droplet_geometry(mask):
    N = mask.shape[0]
    occupied_coords = np.argwhere(mask)
    area = len(occupied_coords)
    padded_mask = np.pad(mask, 1)
    perimeter = int(np.count_nonzero(padded_mask[:, 1:] != padded_mask[:, :-1])
                    + np.count_nonzero(padded_mask[1:, :] != padded_mask[:-1, :]))
    result = dict(droplet_area=area, droplet_area_fraction=area/N**2,
                  droplet_perimeter=perimeter, droplet_perimeter_scaled=perimeter/N,
                  droplet_center=None, horizontal_diameter=0, vertical_diameter=0,
                  radius_of_gyration=None, aspect_ratio=None, compactness=None)
    if area:
        center = occupied_coords.mean(axis=0)
        centered_coords = occupied_coords-center
        covariance = centered_coords.T@centered_coords/area
        eigenvalues = np.linalg.eigvalsh(covariance)
        result.update(droplet_center=(center+1).tolist(),
                      horizontal_diameter=int(np.ptp(occupied_coords[:,1])+1),
                      vertical_diameter=int(np.ptp(occupied_coords[:,0])+1),
                      radius_of_gyration=float(np.sqrt(np.trace(covariance))),
                      aspect_ratio=float(np.sqrt(eigenvalues[1]/eigenvalues[0])) if eigenvalues[0]>0 else None,
                      compactness=float(4*np.pi*area/perimeter**2))
    return result


def peierls_contours(state):
    """Trace closed dual-edge walks, with plus spins on the right.

    Coordinates are in padded-array row/column units (half integers).
    At alternating plaquettes pair N-W and S-E. A walk may revisit a
    vertex, but never an edge; the ordered edges encode the resolution.
    """
    outgoing = {}

    def add_edge(start, end, plus_on_right):
        if not plus_on_right:
            start, end = end, start
        outgoing.setdefault(start, []).append(end)

    # Integer dual coordinates denote the upper-left site of a plaquette.
    for row, col in np.argwhere(state[:, :-1] != state[:, 1:]):
        row, col = int(row), int(col)
        add_edge((row-1, col), (row, col), state[row, col] == 1)
    for row, col in np.argwhere(state[:-1, :] != state[1:, :]):
        row, col = int(row), int(col)
        add_edge((row, col-1), (row, col), state[row+1, col] == 1)

    pairing = {(-1, 0): (0, -1), (0, -1): (-1, 0),
               (1, 0): (0, 1), (0, 1): (1, 0)}
    remaining = {(start, end) for start, ends in outgoing.items() for end in ends}
    contours = []
    while remaining:
        first = next(iter(remaining))
        start, end = first
        vertices = [start]
        while True:
            remaining.remove((start, end))
            vertices.append(end)
            candidates = outgoing[end]
            if len(candidates) == 1:
                following = candidates[0]
            else:
                incoming = (start[0]-end[0], start[1]-end[1])
                dr, dc = pairing[incoming]
                following = (end[0]+dr, end[1]+dc)
            start, end = end, following
            if (start, end) == first:
                break
        contours.append(np.asarray(vertices, dtype=float) + 0.5)
    return contours
