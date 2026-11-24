"""Grand monotone coupling of extremal random-site heat-bath chains."""
import numpy as np
from .model import IsingModel, validate_algorithm
from .random_stream import RandomUpdateStream
from .kernels.local import local_tables


class MonotoneCoupling:
    """Run shared-noise extremal chains until coalescence; time in attempts."""
    def __init__(self, model: IsingModel = IsingModel(), seed: int | None = None, *, backend: str = 'numba'):
        validate_algorithm('heatbath', backend)
        self._model = model
        self._geometry = model.geometry
        self._backend = backend
        self.rng = np.random.default_rng(seed)
        self._lower = self._geometry.make_state('minus', self.rng)
        self._upper = self._geometry.make_state('plus', self.rng)
        self._hb, _ = local_tables(model)
        self.updates = 0
        self._disagreement = model.num_sites
        self._stream = RandomUpdateStream(self._geometry, self.rng)
        self._kernel = self._geometry.coupled_kernel(backend)
        self._kernel_args = self._geometry.local_kernel_args

    @property
    def model(self) -> IsingModel:
        return self._model

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def lower(self):
        view = self._lower.view()
        view.flags.writeable = False
        return view

    @property
    def upper(self):
        view = self._upper.view()
        view.flags.writeable = False
        return view

    @property
    def coalesced(self) -> bool:
        return self._disagreement == 0

    @property
    def t(self) -> float:
        return self.updates/self._geometry.num_sites

    @property
    def disagreement_fraction(self) -> float:
        return self._disagreement/self._geometry.num_sites

    def advance_updates(self, count: int = 1):
        if not isinstance(count, int) or count<0:
            raise ValueError('count must be a nonnegative integer')
        while count and not self.coalesced:
            sites, uniforms = self._stream.peek(count)
            attempts_used, self._disagreement = self._kernel(
                self._lower.ravel(), self._upper.ravel(), *self._kernel_args, sites, uniforms,
                self._hb, self._disagreement)
            self._stream.consume(attempts_used)
            self.updates += attempts_used
            count -= attempts_used
        return self

    def advance(self, sweeps: int = 1):
        if not isinstance(sweeps,int) or sweeps<0:
            raise ValueError('sweeps must be a nonnegative integer')
        return self.advance_updates(sweeps*self._geometry.num_sites)

    def observe(self):
        return dict(updates=self.updates,sweep=self.t,coalesced=self.coalesced,
                    disagreement_fraction=self.disagreement_fraction,
                    lower_magnetization=float(self._geometry.interior(self.lower).mean()),
                    upper_magnetization=float(self._geometry.interior(self.upper).mean()))
