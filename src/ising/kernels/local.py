"""Sequential random-site transitions and shared-noise heat-bath updates."""
import numpy as np
from numba import njit


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


def coupled_updates_python(lower, upper, w, sites, uniforms, hb, disagreement):
    """Apply shared site/uniform pairs, stopping exactly at coalescence."""
    for update_idx in range(len(sites)):
        site = sites[update_idx]
        disagreement -= int(lower[site] != upper[site])
        for flat_spins in (lower, upper):
            num_plus_neighbors = (int(flat_spins[site-1])+int(flat_spins[site+1])+int(flat_spins[site-w])+int(flat_spins[site+w])+4)//2
            flat_spins[site] = 1 if uniforms[update_idx] < hb[num_plus_neighbors] else -1
        disagreement += int(lower[site] != upper[site])
        if disagreement == 0:
            return update_idx+1, disagreement
    return len(sites), disagreement


coupled_updates_numba = njit(cache=True)(coupled_updates_python)
