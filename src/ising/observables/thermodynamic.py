"""Energy and broken bonds, with each interior-incident edge counted once."""
import numpy as np


def magnetization(state):
    return float(state[1:-1, 1:-1].mean())


def interface_length(state):
    return int(np.count_nonzero(state[1:-1, :-1] != state[1:-1, 1:])
               + np.count_nonzero(state[:-1, 1:-1] != state[1:, 1:-1]))


def bond_sum(state):
    """Sum spin products over edges incident to the interior."""
    return int((state[1:-1, :-1]*state[1:-1, 1:]).sum() + (state[:-1, 1:-1]*state[1:, 1:-1]).sum())


def thermodynamics(N, h, spin_sum, length):
    bonds = 2*N*(N+1)
    interaction = 2*length - bonds
    field = -h*spin_sum
    return dict(interface_length=int(length), interaction_energy=interaction,
                field_energy=field, energy=interaction+field,
                excess_energy=2*length-h*(spin_sum+N*N))
