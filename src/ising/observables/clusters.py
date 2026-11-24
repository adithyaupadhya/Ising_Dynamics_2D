"""Northwest-connected geometric plus clusters, not FK clusters."""
import numpy as np
from scipy.ndimage import label
from ..geometry.square import NW_CONNECTIVITY as NW_CONNECTIVITY, interior


def plus_components(state):
    component_labels, _ = label(interior(state) == 1, structure=NW_CONNECTIVITY)
    component_sizes = np.bincount(component_labels.ravel())[1:]
    return component_labels, component_sizes


def cluster_statistics(labels, counts, min_size=1):
    if not isinstance(min_size, int) or min_size < 1:
        raise ValueError('min_size must be a positive integer')
    sorted_sizes = np.sort(counts)[::-1]
    largest_size = int(sorted_sizes[0]) if len(sorted_sizes) else 0
    second_largest_size = int(sorted_sizes[1]) if len(sorted_sizes)>1 else 0
    return dict(cluster_sizes=sorted_sizes.tolist(), cluster_count=len(sorted_sizes),
                clusters_above_threshold=int(np.count_nonzero(sorted_sizes>=min_size)),
                largest_plus_fraction=largest_size/labels.size, second_plus_fraction=second_largest_size/labels.size,
                second_to_first_ratio=second_largest_size/largest_size if largest_size else None,
                cluster_second_moment=float(np.sum(sorted_sizes.astype(float)**2)/labels.size))


def largest_plus_fraction(state):
    _, component_sizes = plus_components(state)
    return float(component_sizes.max(initial=0)/interior(state).size)

