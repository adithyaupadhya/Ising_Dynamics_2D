"""Thermodynamics from physical edge and spin counts; square-state helpers."""
from ..geometry.square import SquareLattice, interior, bond_pairs, broken_bonds


def magnetization(state):
    return float(interior(state).mean())


def interface_length(state):
    return broken_bonds(state)


def bond_sum(state):
    """Sum spin products over edges incident to the interior."""
    return sum(int((a * b).sum()) for a, b in bond_pairs(state))


def thermodynamics_from_counts(num_sites, num_bonds, h, spin_sum, length):
    """Unit-coupling Ising energy, with a uniform field and all-minus reference."""
    interaction = 2*length - num_bonds
    field = -h*spin_sum
    return dict(interface_length=int(length), interaction_energy=interaction,
                field_energy=field, energy=interaction+field,
                excess_energy=2*length-h*(spin_sum+num_sites))


def thermodynamics(N, h, spin_sum, length):
    """Thermodynamics of the square box (existing public signature)."""
    geometry = SquareLattice(N)
    return thermodynamics_from_counts(geometry.num_sites, geometry.num_bonds, h, spin_sum, length)
