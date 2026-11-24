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
