"""Four-connected geometric plus clusters, not FK clusters."""
import numpy as np
from scipy.ndimage import label


def plus_components(state):
    labels, _ = label(state[1:-1, 1:-1] == 1)
    counts = np.bincount(labels.ravel())[1:]
    return labels, counts


def cluster_statistics(labels, counts, min_size=1):
    if not isinstance(min_size, int) or min_size < 1:
        raise ValueError('min_size must be a positive integer')
    sizes = np.sort(counts)[::-1]
    a = int(sizes[0]) if len(sizes) else 0
    b = int(sizes[1]) if len(sizes)>1 else 0
    return dict(cluster_sizes=sizes.tolist(), cluster_count=len(sizes),
                clusters_above_threshold=int(np.count_nonzero(sizes>=min_size)),
                largest_plus_fraction=a/labels.size, second_plus_fraction=b/labels.size,
                second_to_first_ratio=b/a if a else None,
                cluster_second_moment=float(np.sum(sizes.astype(float)**2)/labels.size))


def largest_plus_fraction(state):
    _, counts = plus_components(state)
    return float(counts.max(initial=0)/(state.shape[0]-2)**2)

