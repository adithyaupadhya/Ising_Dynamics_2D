"""Grand monotone coupling of extremal random-site heat-bath chains."""
import numpy as np
from .model import IsingModel, validate_algorithm
from .random_stream import RandomUpdateStream
from .state import make_state
from .kernels.local import local_tables, coupled_updates_python, coupled_updates_numba


class MonotoneCoupling:
    """Run shared-noise extremal chains until coalescence; time in attempts."""
    def __init__(self, model: IsingModel = IsingModel(), seed: int | None = None, *, backend: str = 'numba'):
        validate_algorithm('heatbath', backend)
        self._model = model
        self._backend = backend
        self.rng = np.random.default_rng(seed)
        self._lower = make_state(model.N, 'minus', self.rng)
        self._upper = make_state(model.N, 'plus', self.rng)
        self._hb, _ = local_tables(model)
        self.updates = 0
        self._disagreement = model.N**2
        self._stream = RandomUpdateStream(model.N, self.rng)
        self._kernel = coupled_updates_numba if backend=='numba' else coupled_updates_python

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
        return self.updates/self.model.N**2

    @property
    def disagreement_fraction(self) -> float:
        return self._disagreement/self.model.N**2

    def advance_updates(self, count: int = 1):
        if not isinstance(count, int) or count<0:
            raise ValueError('count must be a nonnegative integer')
        N = self.model.N
        while count and not self.coalesced:
            sites, uniforms = self._stream.peek(count)
            attempts_used, self._disagreement = self._kernel(
                self._lower.ravel(), self._upper.ravel(), N+2, sites, uniforms,
                self._hb, self._disagreement)
            self._stream.consume(attempts_used)
            self.updates += attempts_used
            count -= attempts_used
        return self

    def advance(self, sweeps: int = 1):
        if not isinstance(sweeps,int) or sweeps<0:
            raise ValueError('sweeps must be a nonnegative integer')
        return self.advance_updates(sweeps*self.model.N**2)

    def observe(self):
        return dict(updates=self.updates,sweep=self.t,coalesced=self.coalesced,
                    disagreement_fraction=self.disagreement_fraction,
                    lower_magnetization=float(self.lower[1:-1,1:-1].mean()),
                    upper_magnetization=float(self.upper[1:-1,1:-1].mean()))
