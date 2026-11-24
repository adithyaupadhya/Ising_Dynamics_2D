"""State, random stream, incremental magnetization and simulation clock."""
import numpy as np
from .model import IsingModel, validate_algorithm
from .random_stream import RandomUpdateStream
from .kernels.local import local_tables
from .observables.thermodynamic import thermodynamics_from_counts
from .observables.clusters import cluster_statistics


class Simulator:
    """Local time counts individual attempts; SW time counts cluster steps."""

    def __init__(self, model: IsingModel = IsingModel(), init: str = 'minus', seed: int | None = None, *, dynamics: str = 'heatbath', backend: str = 'numba'):
        validate_algorithm(dynamics, backend)
        self._dynamics = dynamics
        self._backend = backend
        self._model = model
        self._geometry = model.geometry
        self.rng = np.random.default_rng(seed)
        self._state = self._geometry.make_state(init, self.rng)
        self._spin_sum = int(self._geometry.interior(self._state).sum())
        self._length = self._geometry.broken_bonds(self._state)
        self.updates = 0
        self._hb, self._mp = local_tables(model)
        self._kernel = self._geometry.local_kernel(backend)
        self._kernel_args = self._geometry.local_kernel_args
        self._cluster_kernel = self._geometry.cluster_kernel(dynamics, backend)
        self._stream = RandomUpdateStream(self._geometry, self.rng)

    @property
    def model(self) -> IsingModel:
        return self._model

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dynamics(self) -> str:
        return self._dynamics

    @property
    def state(self):
        """Read-only view; use set_state to replace a configuration safely."""
        view = self._state.view()
        view.flags.writeable = False
        return view

    def set_state(self, state):
        state = np.asarray(state)
        self._geometry.validate_state(state)
        self._state[:] = state
        self._spin_sum = int(self._geometry.interior(self._state).sum())
        self._length = self._geometry.broken_bonds(self._state)

    @property
    def t(self) -> float:
        return self.updates if self._cluster_kernel is not None else self.updates / self._geometry.num_sites

    @property
    def magnetization(self) -> float:
        return self._spin_sum / self._geometry.num_sites

    def _advance(self, count, direction=0, threshold=0.0):
        if direction and direction*self.magnetization >= direction*threshold:
            return True
        if self._cluster_kernel is None:
            return self._advance_local(count, direction, threshold)
        return self._advance_cluster(count, direction, threshold)

    def _advance_local(self, count, direction, threshold):
        """Advance sequential attempts, preserving unused random draws."""
        num_sites = self._geometry.num_sites
        while count:
            sites, uniforms = self._stream.peek(count)
            attempts_used, self._spin_sum, self._length = self._kernel(
                self._state.ravel(), *self._kernel_args, sites, uniforms, self._hb, self._mp,
                self.dynamics == 'heatbath', self._spin_sum, self._length,
                direction, threshold, num_sites)
            self._stream.consume(attempts_used)
            self.updates += attempts_used
            count -= attempts_used
            if direction and direction*self.magnetization >= direction*threshold:
                return True
        return False

    def _advance_cluster(self, count, direction, threshold):
        """Advance whole cluster steps and recompute incremental statistics."""
        model = self.model
        while count:
            self._cluster_kernel(self._state, model.beta, model.h, self.rng)
            self._spin_sum = int(self._geometry.interior(self._state).sum())
            self._length = self._geometry.broken_bonds(self._state)
            self.updates += 1
            count -= 1
            if direction and direction*self.magnetization >= direction*threshold:
                return True
        return False

    def advance(self, sweeps: int = 1):
        if not isinstance(sweeps, int) or sweeps < 0:
            raise ValueError('sweeps must be a nonnegative integer')
        self._advance(sweeps * (1 if self._cluster_kernel is not None else self._geometry.num_sites))
        return self

    def advance_updates(self, count: int = 1):
        if not isinstance(count, int) or count < 0:
            raise ValueError('count must be a nonnegative integer')
        self._advance(count)
        return self

    def first_passage(self, limit: int, threshold: float = 0.0, direction: int = 1) -> bool:
        """Advance at most limit sweeps, checking after every elementary update."""
        if not isinstance(limit, int) or limit < 0 or direction not in (-1, 1) or not -1 <= threshold <= 1:
            raise ValueError('Invalid passage parameters')
        return self._advance(limit * (1 if self._cluster_kernel is not None else self._geometry.num_sites), direction, threshold)

    def observe(self, clusters=True, *, thermodynamic=True, cluster_details=False,
                contours=False, boundary_profile=False, interface_heights=False,
                min_cluster_size=1, trim=0):
        """Measure requested quantities; reuse component labels and droplet mask."""
        obs = self._geometry.observables
        magnetization = self.magnetization
        result = dict(sweep=self.t, updates=self.updates, magnetization=magnetization, plus_fraction=(1+magnetization)/2)
        if thermodynamic:
            result.update(thermodynamics_from_counts(
                self._geometry.num_sites, self._geometry.num_bonds,
                self.model.h, self._spin_sum, self._length))
        if clusters or cluster_details or contours or interface_heights:
            component_labels, component_sizes = obs.plus_components(self._state)
            if clusters:
                result['largest_plus_fraction'] = float(component_sizes.max(initial=0)/component_labels.size)
            if cluster_details:
                result.update(cluster_statistics(component_labels, component_sizes, min_cluster_size))
            if contours or interface_heights:
                filled_droplet = obs.droplet_mask(component_labels, component_sizes)
                if contours:
                    result.update(obs.droplet_geometry(filled_droplet))
                if interface_heights:
                    result['interface_heights'] = obs.interface_heights(filled_droplet, trim)
        if boundary_profile:
            result['boundary_magnetization'] = obs.boundary_magnetization(self._state)
        return result
