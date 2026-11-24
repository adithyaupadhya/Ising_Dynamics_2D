"""State, random stream, incremental magnetization and simulation clock."""
import numpy as np
from .model import IsingModel, validate_algorithm
from .random_stream import RandomUpdateStream
from .state import make_state
from .kernels import local_tables, random_updates_python, random_updates_numba, CLUSTER_KERNELS
from . import observables as obs


class Simulator:
    """Local time counts individual attempts; SW time counts cluster steps."""

    def __init__(self, model: IsingModel = IsingModel(), init: str = 'minus', seed: int | None = None, *, dynamics: str = 'heatbath', backend: str = 'numba'):
        validate_algorithm(dynamics, backend)
        self._dynamics = dynamics
        self._backend = backend
        self._model = model
        self.rng = np.random.default_rng(seed)
        self._state = make_state(model.N, init, self.rng)
        self._spin_sum = int(self._state[1:-1, 1:-1].sum())
        self._length = obs.interface_length(self._state)
        self.updates = 0
        self._hb, self._mp = local_tables(model)
        self._kernel = (random_updates_numba if backend == 'numba'
                        else random_updates_python)
        self._cluster_kernel = CLUSTER_KERNELS.get((dynamics, backend))
        self._stream = RandomUpdateStream(model.N, self.rng)

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
        if (state.shape != self._state.shape or not np.isin(state, [-1, 1]).all()
                or not np.all(state[[0, -1], :] == -1)
                or not np.all(state[:, [0, -1]] == -1)):
            raise ValueError('Invalid padded spin configuration')
        self._state[:] = state
        self._spin_sum = int(self._state[1:-1, 1:-1].sum())
        self._length = obs.interface_length(self._state)

    @property
    def t(self) -> float:
        return self.updates if self._cluster_kernel is not None else self.updates / self.model.N**2

    @property
    def magnetization(self) -> float:
        return self._spin_sum / self.model.N**2

    def _advance(self, count, direction=0, threshold=0.0):
        model = self.model
        num_sites = model.N**2
        if direction and direction*self.magnetization >= direction*threshold:
            return True
        while count:
            if self._cluster_kernel is not None:
                self._cluster_kernel(self._state, model.beta, model.h, self.rng)
                self._spin_sum = int(self._state[1:-1, 1:-1].sum())
                self._length = obs.interface_length(self._state)
                self.updates += 1
                count -= 1
            else:
                sites, uniforms = self._stream.peek(count)
                attempts_used, self._spin_sum, self._length = self._kernel(
                    self._state.ravel(), model.N+2, sites, uniforms, self._hb, self._mp,
                    self.dynamics == 'heatbath', self._spin_sum, self._length, direction, threshold, num_sites)
                self._stream.consume(attempts_used)
                self.updates += attempts_used
                count -= attempts_used
            if direction and direction*self.magnetization >= direction*threshold:
                return True
        return False

    def advance(self, sweeps: int = 1):
        if not isinstance(sweeps, int) or sweeps < 0:
            raise ValueError('sweeps must be a nonnegative integer')
        self._advance(sweeps * (1 if self._cluster_kernel is not None else self.model.N**2))
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
        return self._advance(limit * (1 if self._cluster_kernel is not None else self.model.N**2), direction, threshold)

    def observe(self, clusters=True, *, thermodynamic=True, cluster_details=False,
                contours=False, boundary_profile=False, interface_heights=False,
                min_cluster_size=1, trim=0):
        """Measure requested quantities; reuse component labels and droplet mask."""
        magnetization = self.magnetization
        result = dict(sweep=self.t, updates=self.updates, magnetization=magnetization, plus_fraction=(1+magnetization)/2)
        if thermodynamic:
            result.update(obs.thermodynamics(self.model.N, self.model.h, self._spin_sum, self._length))
        if clusters or cluster_details or contours or interface_heights:
            component_labels, component_sizes = obs.plus_components(self._state)
            if clusters:
                result['largest_plus_fraction'] = float(component_sizes.max(initial=0)/component_labels.size)
            if cluster_details:
                result.update(obs.cluster_statistics(component_labels, component_sizes, min_cluster_size))
            if contours or interface_heights:
                filled_droplet = obs.droplet_mask(component_labels, component_sizes)
                if contours:
                    result.update(obs.droplet_geometry(filled_droplet))
                if interface_heights:
                    result['interface_heights'] = obs.interface_heights(filled_droplet, trim)
        if boundary_profile:
            result['boundary_magnetization'] = obs.boundary_magnetization(self._state)
        return result
