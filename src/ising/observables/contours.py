"""Lattice droplet proxy: fill holes in the largest four-connected plus cluster."""
import numpy as np
from scipy.ndimage import binary_fill_holes


def droplet_mask(labels, counts):
    if not len(counts):
        return np.zeros(labels.shape, dtype=bool)
    # Eight-connected background avoids closing diagonal escape channels.
    return binary_fill_holes(labels == (int(np.argmax(counts))+1),
                             structure=np.ones((3, 3), dtype=bool))


def droplet_geometry(mask):
    n = mask.shape[0]
    coords = np.argwhere(mask)
    area = len(coords)
    padded = np.pad(mask, 1)
    perimeter = int(np.count_nonzero(padded[:, 1:] != padded[:, :-1])
                    + np.count_nonzero(padded[1:, :] != padded[:-1, :]))
    result = dict(droplet_area=area, droplet_area_fraction=area/n**2,
                  droplet_perimeter=perimeter, droplet_perimeter_scaled=perimeter/n,
                  droplet_center=None, horizontal_diameter=0, vertical_diameter=0,
                  radius_of_gyration=None, aspect_ratio=None, compactness=None)
    if area:
        center = coords.mean(axis=0)
        centered = coords-center
        covariance = centered.T@centered/area
        eigenvalues = np.linalg.eigvalsh(covariance)
        result.update(droplet_center=(center+1).tolist(),
                      horizontal_diameter=int(np.ptp(coords[:,1])+1),
                      vertical_diameter=int(np.ptp(coords[:,0])+1),
                      radius_of_gyration=float(np.sqrt(np.trace(covariance))),
                      aspect_ratio=float(np.sqrt(eigenvalues[1]/eigenvalues[0])) if eigenvalues[0]>0 else None,
                      compactness=float(4*np.pi*area/perimeter**2))
    return result
