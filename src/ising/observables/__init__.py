"""Public observable API. Geometric operations are opt-in beyond the largest cluster."""
from .thermodynamic import magnetization, bond_sum, interface_length, thermodynamics
from ..geometry.square import NW_CONNECTIVITY
from .clusters import largest_plus_fraction, plus_components, cluster_statistics
from .contours import droplet_mask, droplet_geometry, peierls_contours
from .profiles import boundary_magnetization, interface_heights

__all__ = ["magnetization", "bond_sum", "interface_length", "thermodynamics",
           "largest_plus_fraction", "plus_components", "cluster_statistics",
           "NW_CONNECTIVITY", "peierls_contours", "droplet_mask", "droplet_geometry", "boundary_magnetization", "interface_heights"]
