from __future__ import annotations

import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from typeguard import typechecked

from .history import HistoryTracker
from .integrator import Integrator, IVPIntegrator
from .utils import (
    allowed_kwargs_for_func,
    mean_vector_to_ensemble,
    normalized_alpha,
    normalized_time,
    normalized_y,
)

__all__ = ['Model']



# %% =================================== PARENT MODEL CLASS ============================================= %% #
class Model:
    r"""Base class for all forecast models.

    A `Model` couples three ingredients:

    - a **state history** ([`HistoryTracker`][dynamodels.history.HistoryTracker])
      with pre-allocated storage of shape $(N_t,\, N_\phi\,[+N_\alpha],\, m)$ — time,
      physical state (plus estimated parameters, once `init_ensemble` augments them),
      and ensemble members;
    - a **time-integration strategy**
      ([`Integrator`][dynamodels.integrator.Integrator]) selected at construction;
    - the **observation operator** `M`, used by ensemble estimators to map the
      *analysis-augmented* state (state, estimated parameters, and observables
      stacked, size $N=N_\phi+N_\alpha+N_q$) to the observables.

    Physical models implement ``time_derivative(t, psi, **params)`` (continuous) or
    ``time_step(Nt)`` (discrete maps) and declare their estimable parameters in
    ``params`` with bounds in ``alpha_lims``. By convention, the leading $N_q$
    components of the physical state are directly observable (see
    `get_observables`).

    Parameters
    ----------
    psi0 : np.ndarray or list
        Initial state, shape $(N_\phi,)$ or $(N_\phi, m)$.
    dt : float
        Output time step.
    integrator_class : type[Integrator]
        Time-integration strategy (default `IVPIntegrator`).
    **kwargs
        Model-parameter overrides (any attribute defined by the child class).

    Attributes
    ----------
    params : list of str
        Names of the parameters that can be varied/estimated; declared by child
        classes.
    fixed_params : list of str
        Names of parameters treated as fixed and forwarded to the governing
        equations through `governing_eqns_params` (see `set_fixed_params`).
    extra_print_params : list of str
        Extra attribute names appended to `params` when building `print_params`.
    governing_eqns_params : dict
        Extra keyword arguments passed to `time_derivative` / `time_step`.
    t_transient : float
        Transient time discarded before the pre-allocated history / ensemble
        generation starts (see `init_ensemble`, `create_long_timeseries`).
    t_CR : float
        Characteristic (e.g. recurrence) time used to size the "zoom" window in
        the `visualize_*_hist` plots.
    Nq : int
        Number of observable components; declared by child classes (default 1).
    alpha : dict or None
        Copy of `alpha0` taken at construction; not updated automatically as
        parameters evolve (use `get_alpha` for per-member current values).
    initialized : bool
        Set to True once `__init__` has completed.
    results_folder : str or None
        Optional path for saving results.
    """

    params = []  # List of parameter names that can be varied in the model
    fixed_params = []
    extra_print_params = []
    governing_eqns_params = dict()

    t = 0.
    t_transient = 0.
    t_CR = 10 * 0.01


    Nq = 1
    alpha = None

    initialized = False
    results_folder = None

    @typechecked
    def __init__(self,
                 psi0: np.ndarray | list,
                 dt: float,
                 integrator_class: type[Integrator] = IVPIntegrator,
                 **kwargs):

        # ================= INITIALISE PHYSICAL MODEL ================== ##
        keys = list(kwargs.keys())
        [setattr(self, key, kwargs.pop(key)) for key in keys if hasattr(self, key)]

        if len(kwargs.keys()) > 1:
            print(f'Model key(s) {kwargs.keys()} not assigned')

        # ====================== SET INITIAL CONDITIONS ====================== ##

        # Ensure psi0 is ndarray with ndim=2
        if psi0 is None:
            raise ValueError("Initial state psi0 must be provided during Model initialization.")
        elif (isinstance(psi0, np.ndarray) and psi0.ndim == 1) or isinstance(psi0, list):
            psi0 = np.array([psi0]).T

        self.psi0 = psi0
        self.dt = dt
        self.alpha0 = {par: getattr(self, par) for par in self.params}
        self.alpha = self.alpha0.copy()

        # ========================== CREATE HISTORY ========================== ##
        # self._initial_capacity = int(self.t_transient / self.dt) # Initial capacity of history arrays
        # self._current_ti = 1  # Current time index in history arrays

        self.history = HistoryTracker()
        self.history._initial_capacity = int(self.t_transient / self.dt)*2 if self.t_transient > 0 else 1000
        self.update_history(psi=self.psi0[np.newaxis, :, :],
                            t=np.array([0.]),
                            reset=True)

        # ======================== SET RNG ================================== ##
        self.print_params = self.define_print_params()
        self.set_fixed_params()
        self.initialized = True

        # ================= INITIALISE INTEGRATOR STRATEGY ================== ##
        # The model holds an instance of the specific Integrator
        self.integrator = integrator_class(self)


    def update_history(self, psi: np.ndarray, t=None, reset=False, modify_saved_states=False):
        r"""Append (or overwrite) states in the model's `history`.

        Parameters
        ----------
        psi : ndarray
            State(s) to store; reshaped to $(N_t, N_\phi\,[+N_\alpha], m)$ if
            given with fewer dimensions.
        t : ndarray or float, optional
            Time stamp(s) matching `psi`. If None and neither `reset` nor
            `modify_saved_states` is set, it is inferred as
            ``current_time + arange(Nt) * dt``.
        reset : bool
            If True, replace the entire history with `psi` (and restart the
            clock unless `t` is given).
        modify_saved_states : bool
            If True, overwrite the most recent ``psi.shape[0]`` entries already
            in history in place (e.g. after a data-assimilation analysis step)
            instead of appending new ones.
        """
        psi = self.__format_state(psi)
        if t is None and not reset and not modify_saved_states:

            t = (np.arange(0, psi.shape[0]) * self.dt).round(self.precision_t) + self.current_time
        if isinstance(t, float):
            t = np.array([t])

        # if modify_saved_states:
            # print(f"Updating history with new state of shape {psi.shape} and time array {t}. \
            #     Reset={reset}, modify_saved_states={modify_saved_states}")

        self.history.update_history(psi, t=t, reset=reset, modify_saved_states=modify_saved_states)




    @property
    def state_labels(self):
        r"""list of str: LaTeX labels $\phi_0, \phi_1, \dots$ for each physical
        state component, used by the `visualize_*` helpers."""
        return  [f'$\\phi_{{{kk}}}$' for kk in range(self.Nphi)]

    @property
    def obs_labels(self):
        """list of str: LaTeX labels for the observable components.

        Must be implemented by child classes; raises `NotImplementedError` here.
        """
        raise NotImplementedError("obs_labels property must be implemented in the child class.")


    @property
    def name(self):
        """str: Model name used e.g. in filenames and plot titles.

        Defaults to the class name if not explicitly set.
        """
        return getattr(self, '_name', self.__class__.__name__)

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def alpha_lims(self):
        """dict: Mapping ``{param_name: (lower, upper)}`` of physical bounds for
        each entry in `params`.

        Lazily initialised to ``(None, None)`` (unbounded) for every parameter
        the first time it is accessed.
        """
        if not hasattr(self, '_alpha_lims'):
            self._alpha_lims = {key: (None, None) for key in sorted(self.params)}

        return self._alpha_lims

    @alpha_lims.setter
    def alpha_lims(self, value: dict):
        """Update `alpha_lims`.

        Parameters
        ----------
        value : dict
            Mapping ``{param_name: (lower, upper)}``; keys must be a subset of
            `params`. Merged with any existing bounds; parameters not present in
            `value` keep (or default to) ``(None, None)``.
        """
        assert set(value.keys()) - set(self.params) == set(), f"Keys of alpha_lims must be a subset of {self.params}, but got {value.keys()}"
        if hasattr(self, '_alpha_lims'):
            self._alpha_lims.update(value)
        else:
            self._alpha_lims = value
            if len(self._alpha_lims) < len(self.params):
                missing_keys = set(self.params) - set(self._alpha_lims.keys())
                self._alpha_lims.update({key: (None, None) for key in missing_keys})


    @property
    def alpha_labels(self):
        r"""dict: Default parameter-label mapping.

        Lazily initialised, the first time it is accessed, to
        ``{name_0: '$\alpha_0$', name_1: '$\alpha_1$', ...}`` — one entry per
        name in `params` (alphabetically sorted), keyed by parameter name. Assign
        a custom mapping via the setter, using the same key convention.
        """
        if not hasattr(self, '_alpha_labels'):
            self._alpha_labels = {val: f'$\\alpha_{ii}$' for ii, val in enumerate(sorted(self.params))}
        return self._alpha_labels

    @alpha_labels.setter
    def alpha_labels(self, value: dict):
        """Update `alpha_labels`.

        Parameters
        ----------
        value : dict
            Keyed by parameter name (a subset of `params`), same convention as the
            getter. Merged with any existing labels; parameters missing from
            `value` are back-filled with an auto-generated LaTeX label.
        """
        assert set(list(value.keys())) - set(sorted(self.params)) == set(), f"Keys of alpha_labels must be a subset of {sorted(self.params)}, but got {value.keys()}"
        if hasattr(self, '_alpha_labels'):
            self._alpha_labels.update(value)
        else:
            self._alpha_labels = value
            if len(self._alpha_labels) < len(self.params):
                missing_keys = set(sorted(self.params)) - set(self._alpha_labels.keys())
                self._alpha_labels.update({val: f'$\\alpha_{ii}$' for ii, val in enumerate(missing_keys)})


    @property
    def hist(self):
        """Returns only the valid (non-empty) portion of the history buffer."""
        return self.history.hist

    @property
    def hist_t(self):
        """Returns only the valid portion of the time history."""
        return self.history.hist_t

    @property
    def current_state(self):
        r"""ndarray: Most recent state in `history`, shape $(N_\phi\,[+N_\alpha], m)$."""
        return self.history.current_state

    @property
    def current_time(self):
        """float: Time stamp of `current_state`."""
        return self.history.current_time

    @property
    def filename(self):
        """str: Descriptive filename built from `name` and the parameters in
        `alpha0` that differ from their class defaults (falls back to
        ``f"{name}_default"`` if none differ). Cached after first access; also
        extended with an ``_ensemble_m{m}`` suffix by `init_ensemble`.
        """
        if not hasattr(self, '_filename'):
            suffix = ''
            for key, val in self.alpha0.items():
                if val != getattr(self.__class__, key):
                    if np.log10(abs(val)) < -3:
                        suffix += f'_{key}{val:.2e}'
                    else:
                        suffix += f'_{key}{val}'
            if len(suffix) == 0:
                suffix = '_default'
            # Structural parameters (`fixed_params`, e.g. Lorenz96's Nx) are keyed in
            # too, so e.g. the Nx=10 and Nx=40 systems never share a file. Only those
            # with a scalar class default participate: the instance-computed ones
            # (Rijke's collocation arrays, tau_adv) would rename every file.
            for key in self.fixed_params:
                default, val = getattr(type(self), key, None), getattr(self, key)
                if np.isscalar(default) and np.isscalar(val) and val != default:
                    suffix += f'_{key}{val}'
            self._filename = f"{self.name}{suffix}"

        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value


    def __format_state(self, psi: np.ndarray) -> np.ndarray:
        """Ensure `psi` has the 3-D shape expected by `history`.

        Parameters
        ----------
        psi : ndarray
            State array with 1, 2, or 3 dimensions.

        Returns
        -------
        ndarray
            `psi` promoted to shape ``(Nt, N, m)``: ``(N,) -> (1, N, 1)``,
            ``(N, m) -> (1, N, m)``, ``(Nt, N, m)`` unchanged.
        """
        if psi.ndim == 1:
            psi = psi[np.newaxis, :, np.newaxis]  # (N,) -> (1, N, 1)
        elif psi.ndim == 2:
            psi = psi[np.newaxis, :, :]  # (N, m) -> (1, N, m)
        elif psi.ndim == 3:
            pass  # Already in correct shape (Nt, N, m)
        else:
            raise ValueError(f"State array psi has invalid number of dimensions: {psi.ndim}={psi.shape}. Expected 1, 2, or 3.")
        return psi

    def define_print_params(self):
        """list of str: Parameter names shown by `print_parameters` — `params`
        followed by `extra_print_params`.
        """
        return [*self.params, *self.extra_print_params]

    @staticmethod
    def t_lyap_from_table(value, table, fallback):
        """Parameter-dependent Lyapunov time from a table of measured exponents.

        `table` maps sweep-parameter values to the dominant Lyapunov exponent
        $\\lambda_1$ measured there (chaotic points only). Inside the tabulated
        range, returns $1/\\lambda_1$ with $\\lambda_1$ log-interpolated at
        `value`; outside it (limit cycles, tori, untabulated configurations)
        returns `fallback`, where a Lyapunov time is not defined.
        """
        keys = sorted(table)
        if keys[0] <= value <= keys[-1]:
            return 1.0 / float(np.exp(np.interp(value, keys, np.log([table[k] for k in keys]))))
        return fallback

    @property
    def psi0(self):
        r"""ndarray: Initial state/ensemble passed at construction, shape
        $(N_\phi, m)$ (see the `psi0` constructor parameter).

        Re-assigning `psi0` after construction is unusual and emits a
        `UserWarning`.
        """
        return self._psi0

    @psi0.setter
    def psi0(self, value):
        """Re-assign `psi0` (emits `UserWarning`; not generally recommended)."""
        if hasattr(self, '_psi0'):
            warnings.warn(f"psi0 is being re-assigned. Previous shape {self._psi0.shape},"
                          f" new shape {np.array(value).shape}. This is not recommended.", UserWarning)

            if isinstance(value, np.ndarray) and value.ndim == 1:
                value = np.array([value]).T
            self._psi0 = np.array(value)

        self._psi0 = np.array(value)

    @property
    def alpha0(self):
        """dict: Initial parameter values ``{name: value}``, one entry per name
        in `params`, captured at construction time and never mutated afterwards.
        """
        return self._alpha0

    @alpha0.setter
    def alpha0(self, dict_params):
        """Set `alpha0`. Read-only after construction — raises `AttributeError`
        if called again.
        """
        if hasattr(self, '_alpha0'):
            raise AttributeError("alpha0 is read-only and cannot be modified after initialization.")
        self._alpha0 = dict_params

    @property
    def dt(self):
        """float: Output time step, rounded to `precision_t` decimal places."""
        return self._dt


    @dt.setter
    def dt(self, value):
        """Set `dt`, deriving `precision_t` from it as
        ``ceil(-log10(dt) + 2)``.

        Raises
        ------
        ValueError
            If `value` is not strictly positive.
        """
        if value <= 0:
            raise ValueError("Time step must be positive.")
        self._precision_t = int(np.ceil(-np.log10(value) + 2))  # Set precision based on dt
        # print(f'Setting time step dt={value} with precision_t={self._precision_t}')
        self._dt = np.round(value, self.precision_t)

    @property
    def precision_t(self):
        """int: Number of decimal places used to round time stamps, derived
        from `dt` (set as a side effect of the `dt` setter).
        """
        if not hasattr(self, '_precision_t'):
            if not hasattr(self, '_dt'):
                raise AttributeError("dt must be set before accessing precision_t.")
        return self._precision_t


    @property
    def dt_step(self):
        """float: Time step used internally by the integrator.

        Equal to `dt` for models whose output and integration steps coincide;
        discrete models with distinct output/integration steps override this
        (see `DiscreteIntegrator`).
        """
        return self.dt

    @property
    def Nphi(self):
        """int: Number of physical state components, ``len(psi0)``."""
        return len(self.psi0)

    @property
    def Na(self) -> int:
        """int: Number of parameters currently augmented into the state — the
        length of `est_alpha` if an ensemble is configured, else 0.
        """
        if isinstance(self.ensemble_cfg, dict):
            return len(self.ensemble_cfg.get('est_alpha', []))
        return 0

    @property
    def N(self) -> int:
        r"""int: Size $N=N_\phi+N_\alpha+N_q$ of the analysis-augmented state
        vector (state, estimated parameters, and observables stacked), used
        e.g. to size the observation operator `M`. This is *not* the shape of
        the stored `hist` array, which only carries the $N_\phi\,[+N_\alpha]$
        state/parameter rows.
        """
        return self.Nphi + self.Na + self.Nq

    @property
    def m(self):
        """int: Ensemble size — the last (member) dimension of `hist`."""
        return self.hist.shape[-1]


    def set_fixed_params(self):
        """Build the instance-level `governing_eqns_params` used by
        `time_derivative` / `time_step`.

        Collects the current value of every attribute named in `fixed_params`
        and merges it into a fresh instance dict (copied from the class-level
        `governing_eqns_params`, which is left untouched so fixed parameters
        do not leak across `Model` subclasses). Called once during `__init__`.
        """
        fixed_params = dict((key, getattr(self, key)) for key in self.fixed_params)
        # Create an instance-level dict: the class-level default must not be mutated,
        # otherwise fixed parameters leak across different Model subclasses.
        self.governing_eqns_params = {**self.governing_eqns_params, **fixed_params}


    def create_long_timeseries(self, Nt=None):
        """Integrate the model forward and append the result to `history`.

        Useful e.g. to run a model onto its attractor before further use.

        Parameters
        ----------
        Nt : int, optional
            Number of forecast steps. Defaults to ``10 * t_transient / dt``.
        """
        if Nt is None:
            Nt = int(self.t_transient * 10 / self.dt)
        state, t = self.time_integrate(Nt=Nt)
        self.update_history(state, t)
        self.close()


    @property
    def rng(self):
        """numpy.random.Generator: Random-number generator, lazily created
        from `seed` via `numpy.random.default_rng`.
        """
        if not hasattr(self, '_rng'):
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    @property
    def seed(self):
        """int: Seed used to (re)create `rng`. Defaults to 0."""
        if not hasattr(self, '_seed'):
            self._seed = 0
        return self._seed

    @seed.setter
    def seed(self, value: int):
        """Set `seed` and invalidate the cached `rng` so it is rebuilt on next
        access.
        """
        self._seed = value
        if hasattr(self, '_rng'):
            del self._rng


    def copy(self):
        """Model: A deep copy of this model (`copy.deepcopy`)."""
        return deepcopy(self)


    def get_observables(self, Nt=1, **kwargs):
        """Return the most recent observable(s) from `history`.

        By convention, the observables are the leading `Nq` rows of the
        physical state (see `state_labels` / `obs_labels`).

        Parameters
        ----------
        Nt : int
            Number of trailing time steps to return. If 1 (default), the
            leading ``Nt`` axis is dropped.

        Returns
        -------
        ndarray
            Shape ``(Nq, m)`` if ``Nt == 1``, else ``(Nt, Nq, m)``.
        """
        if Nt == 1:
            return self.hist[-1, :self.Nq, :]
        else:
            return self.hist[-Nt:, :self.Nq, :]

    def get_observable_hist(self, Nt=0, **kwargs):
        """Alias for `get_observables` with a different default.

        Parameters
        ----------
        Nt : int
            Number of trailing time steps to return. With the default 0,
            ``hist[-0:]`` is the *full* array, so the entire observable
            history is returned.

        Returns
        -------
        ndarray
            Shape ``(Nt, Nq, m)`` (or ``(Nq, m)`` if ``Nt == 1``); see
            `get_observables`.
        """
        return self.get_observables(Nt, **kwargs)


    def print_parameters(self, show_header=True, indent=0):
        """Print the model class, its `print_params` values, and — if
        configured — `ensemble_cfg`.

        Parameters
        ----------
        show_header : bool
            If True, print a section header first.
        indent : int
            Number of leading spaces for each printed line.
        """
        if show_header:
            print('\n ------------------ Model Parameters ------------------ ')
        print(f'{" " * indent}Model class: {self.__class__.__name__}')
        for key in sorted(self.print_params):
            val = getattr(self, key)
            print(f'{" " * indent}{key} = {val:.6f}' if isinstance(val, float) else f'{" " * indent}{key} = {val}')
        if self.ensemble_cfg is not False:
            print(f'{" " * indent}Ensemble configuration: {self.ensemble_cfg}')

    # --------------------- DEFINE OBS-STATE MAP --------------------- ##

    @property
    def M(self):
        r"""ndarray or callable: Linear observation operator, shape $(N_q, N)$
        with $N=N_\phi+N_\alpha+N_q$.

        Used by ensemble estimators to extract the observables from the
        analysis-augmented state $[\phi;\alpha;y]$ (state, estimated
        parameters, and observables stacked), $\mathbf{y}=\mathbf{M}\psi$.
        Lazily initialised to the default block matrix
        $[\mathbf{0}_{N_q\times(N_\phi+N_\alpha)},\ \mathbb{I}_{N_q}]$, which
        selects the trailing $N_q$ rows — consistent with the observables
        being appended at the bottom of that augmented vector.
        """
        if not hasattr(self, '_M'):
            self.M = None # This will trigger the setter to create the default M matrix
        return self._M

    @M.setter
    def M(self, M=None):
        r"""Set `M`.

        Parameters
        ----------
        M : ndarray, shape $(N_q, N)$, optional
            Custom observation operator. If None (default), the block matrix
            described in the getter's docstring is (re)built.
        """
        if M is None:
            # M matrix is constructed by horizontally stacking a zero matrix of shape (Nq, Na + Nphi)
            # and an identity matrix of shape (Nq, Nq)
            M = np.hstack((np.zeros([self.Nq, self.Na + self.Nphi]),
                           np.eye(self.Nq)))
        else:
            assert M.shape == (self.Nq, self.N), f"Shape of M must be ({self.Nq, self.N}), but got {M.shape}"

        self._M = M


    @property
    def Ma(self):
        r"""ndarray: Parameter observation operator, shape $(N_\alpha, N)$.

        Selects the $N_\alpha$ estimated-parameter rows from the
        analysis-augmented state $[\phi;\alpha;y]$: the block matrix
        $[\mathbf{0}_{N_\alpha\times N_\phi},\ \mathbb{I}_{N_\alpha},\
        \mathbf{0}_{N_\alpha\times N_q}]$.
        """
        if not hasattr(self, '_Ma'):
            self._Ma = np.hstack((np.zeros([self.Na, self.Nphi]),
                                            np.eye(self.Na),
                                            np.zeros([self.Na, self.Nq])))
        return self._Ma

    # ------------------------- Functions for update/initialise the model --------------------------- #

    def reset_model(self, psi0=None, **kwargs):
        """Re-initialise this model in place via `Model.__init__`.

        Parameters
        ----------
        psi0 : ndarray, optional
            New initial state; defaults to `current_state`.
        **kwargs
            Forwarded to `Model.__init__` (e.g. `dt`, parameter overrides).
        """
        if psi0 is None:
            psi0 = self.current_state

        Model.__init__(self, psi0=psi0, **kwargs)


    def modify_settings(self, **kwargs):
        """Hook for child classes to adjust internal configuration after
        `ensemble_cfg` changes (called by `init_ensemble`). No-op by default.
        """
        pass


    def close(self):
        """Release resources held by the integrator (e.g. multiprocessing
        pools); delegates to `Integrator.close`.
        """
        self.integrator.close()

    @property
    def ensemble_cfg(self) -> dict | bool:
        """dict or False: Ensemble configuration ``{'est_alpha': [...], 'm': m}``
        set by `init_ensemble`, or False if no ensemble has been configured.
        """
        return getattr(self, '_ensemble_config', False)

    @ensemble_cfg.setter
    def ensemble_cfg(self, config: dict):
        """Setter for ensemble configuration."""
        self._ensemble_config = config


    @property
    def est_alpha(self):
        """list of str: Names of the parameters currently estimated (augmented
        into the state), or ``[]`` if no ensemble is configured.
        """
        if isinstance(self.ensemble_cfg, dict):
            return self.ensemble_cfg.get('est_alpha', [])
        else:
            return []

    @est_alpha.setter
    def est_alpha(self, value):
        """Set `est_alpha`. Warns and does nothing if no ensemble is
        configured (`ensemble_cfg` is not a dict).
        """
        if not isinstance(self.ensemble_cfg, dict):
            warnings.warn("Cannot set est_alpha when ensemble is not configured.")
        else:
            self._ensemble_config['est_alpha'] = value


    def get_alpha(self, psi=None):
        """Build the per-member parameter dict(s) for `psi`.

        Parameters
        ----------
        psi : ndarray, optional
            State to read parameters from; defaults to `current_state`. If it
            has exactly `Nphi` rows (no augmented parameters), every member
            gets a copy of `alpha0` unchanged.

        Returns
        -------
        list of dict
            One ``{name: value}`` dict per ensemble member (length
            ``psi.shape[-1]``): `alpha0` overridden, for each member, by the
            values of its last `Na` state rows at the `est_alpha` names.
        """
        if psi is None:
            psi = self.current_state

        if psi.shape[0] == self.Nphi:
            # print('using the same get_alpha')
            return [self.alpha0.copy()] * psi.shape[-1]

        # ensure psi has members on last axis
        if psi.ndim == 1:
            psi = psi[:, np.newaxis]

        alpha_list = []
        for mi in range(psi.shape[-1]):
            alph = self.alpha0.copy()
            alph.update(zip(self.est_alpha, psi[-self.Na:, mi]))
            alpha_list.append(alph)

        return alpha_list


    # ================= Main Time Integration Method ================= #

    def time_integrate(self, Nt=100, averaged=False):
        r"""Forecast the model `Nt` steps ahead.

        Delegates to the currently configured `Integrator` strategy
        (`self.integrator.advance`); this is just a thin wrapper, kept
        overridable in case a child `Model` needs special handling.

        Parameters
        ----------
        Nt : int
            Number of forecast steps.
        averaged : bool
            Only affects ensemble runs (`IVPIntegrator.advance_ensemble`): if
            False (default), each ensemble member is forecast individually
            with its own parameters (`get_alpha`); if True, only the ensemble
            *mean* trajectory is integrated and each member's deviation from
            the mean is left unchanged (frozen) rather than propagated.

        Returns
        -------
        psi : ndarray, shape $(N_t, N_\phi\,[+N_\alpha], m)$
            Forecasted state, excluding the current (initial) state already
            present in `history`.
        t : ndarray, shape $(N_t,)$
            Time stamps of `psi`.
        """
        return self.integrator.advance(Nt=Nt, averaged=averaged, alpha=self.get_alpha())


    # ══════════════════════════════════════════════════════════════════════════
    # Ensemble initialisation
    # ══════════════════════════════════════════════════════════════════════════

    def init_ensemble(
        self,
        m: int,
        std_phi: float = 0.001,
        std_alpha=0.001,
        est_alpha: list = [],
        distribution_phi: str = "normal",
        distribution_alpha: str = "uniform",
        ensure_mean_at_init: bool = False,
        ensemble_psi0=None,
    ):
        """Generate (or validate) the augmented initial ensemble.

        Generates state and (optionally) parameter uncertainty, stacks them
        into the augmented ensemble psi0 of shape ``(1, Nphi+Na, m)``, stores
        it in the model history, and updates the model filename.

        Parameters
        ----------
        m : int
            Ensemble size.
        std_phi : float
            Fractional std for state perturbations.
        std_alpha : float or dict
            Std (or {name: std}) for parameter perturbations.
        est_alpha : list[str], optional
            Names of parameters to augment into the state.
            If empty, and std_alpha is a dict, all parameters in std_alpha are estimated. If empty and std_alpha is not a dict, no parameters are estimated.
        distribution_phi : str
            Sampling distribution for state ("normal" or "uniform").
        distribution_alpha : str
            Sampling distribution for parameters.
        ensure_mean_at_init : bool
            If True one member is forced to equal the mean.
        ensemble_psi0 : ndarray (Nphi+Na, m) or (1, Nphi+Na, m), optional
            Pre-built ensemble; bypasses generation if provided.

        Notes
        -----
        Does not return a value; the generated (or validated) ensemble is
        written directly to `history` via `update_history(reset=True)`.
        """


        if isinstance(std_alpha, dict):
            est_alpha = sorted(list(std_alpha.keys()))
            # mean_vector_to_ensemble iterates std_alpha.values(), so its row order is the
            # dict's insertion order. est_alpha is sorted, and everything downstream
            # (get_alpha, alpha_limits_matrix, plotting) indexes rows by est_alpha position.
            # Re-key in est_alpha order so the two agree.
            std_alpha = {key: std_alpha[key] for key in est_alpha}


        # Push ensemble config so Na, est_alpha properties resolve correctly.
        self.ensemble_cfg = {'est_alpha': est_alpha.copy(), 'm': m}

        self.modify_settings()  # Allow child classes to modify settings based on the new ensemble configuration.

        # Invalidate M cache so it is rebuilt with the updated N = Nphi + Na + Nq.
        if hasattr(self, '_M'):
            del self._M

        if ensemble_psi0 is None:

            #forecast to avoid initializing before the attractor, which can cause issues for some models (e.g., Lorenz63)
            Ntransient = int(self.t_transient / self.dt)
            if Ntransient > 0:
                psi, t = self.time_integrate(Nt=Ntransient, averaged=False)
                mean_phi0 = np.mean(psi[-1], axis=-1)
            else:
                # no transient (e.g. LinearModel): seed from the current state
                mean_phi0 = np.mean(self.current_state[:self.Nphi], axis=-1)


            psi0 = mean_vector_to_ensemble(
                self.rng, mean_phi0, std_phi, m,
                method=distribution_phi,
                ensure_mean_at_init=ensure_mean_at_init,
            )

            if self.est_alpha:
                mean_a = np.array([getattr(self, a) for a in self.est_alpha])

                # print(f"Generating initial ensemble for parameters {self.est_alpha} with std {std_alpha} and distribution {distribution_alpha}.")
                # print(f"Parameter means shape: {mean_a.shape}")
                alpha0 = mean_vector_to_ensemble(
                    self.rng, mean_a, std_alpha, m,
                    method=distribution_alpha,
                    ensure_mean_at_init=ensure_mean_at_init,
                )
                psi0 = np.concatenate((psi0, alpha0), axis=0)

            ensemble_psi0 = psi0[np.newaxis, :, :]  # (1, Nphi+Na, m)

        else:
            if ensemble_psi0.ndim == 2:
                ensemble_psi0 = ensemble_psi0[np.newaxis, :, :]
            assert ensemble_psi0.shape[-1] == m, (
                f"ensemble_psi0 has {ensemble_psi0.shape[-1]} members, expected {m}."
            )
            assert ensemble_psi0.shape[1] == self.Nphi + self.Na, (
                f"ensemble_psi0 state size {ensemble_psi0.shape[1]}, "
                f"expected {self.Nphi + self.Na}."
            )

        self.update_history(psi=ensemble_psi0, t=self.hist_t[[0]], reset=True)

        self.filename += f"_ensemble_m{m}"
        # print(
        #     f"OK: Initialised {self.filename} history "
        #     f"shape={self.hist.shape}  t={self.hist_t}"
        # )


    # ============================== Visualization methods ============================== #


    def visualize_history(self, **kwargs) -> None:
        """Plot observable and parameter histories.

        Calls `visualize_state_hist`, `visualize_observable_hist`, and
        `visualize_spatiotemporal_hist` in turn, forwarding to each only the
        keyword arguments it accepts.
        """

        for func in [self.visualize_state_hist,
                     self.visualize_observable_hist,
                     self.visualize_spatiotemporal_hist]:
            kwargs_obs = allowed_kwargs_for_func(func, kwargs)
            func(**kwargs_obs)


    def visualize_config(self):
        """No-op hook for subclasses to plot model-specific configuration
        (e.g. spatial mesh, filter kernels)."""
        pass



    def visualize_state(self, **kwargs) -> None:
        """Plot ensemble state (and parameter) distributions via
        `plot_state_distribution`.

        No-op (prints a message) if no ensemble is configured
        (`ensemble_cfg` is False).
        """
        if self.ensemble_cfg is False:
            print("No ensemble configuration found. Cannot plot state distribution.")
            return
        kwargs_state = allowed_kwargs_for_func(plot_state_distribution, kwargs)
        plot_state_distribution(self, **kwargs_state)



    def visualize_state_hist(self, psi=None, t=None, max_modes=10, t_zoom=None,
                             reference_y=1.0, reference_t: float = 1.0):
        """Plot the time evolution of each physical state component.

        Two panels per component: the full history, and a zoomed-in view of
        the last `t_zoom` steps. Complex-valued states get separate real/imag
        traces.

        Parameters
        ----------
        psi : ndarray, optional
            State history to plot; defaults to ``hist[:, :Nphi]``.
        t : ndarray, optional
            Matching time stamps; defaults to the tail of `hist_t`.
        max_modes : int
            Maximum number of state components to plot.
        t_zoom : int, optional
            Number of trailing steps shown in the zoomed panel; defaults to
            ``t_CR / dt``.
        reference_y : float
            Reference value used to normalise `psi` (and its axis label).
        reference_t : float
            Reference value used to normalise `t` (and its axis label).
        """
        if psi is None:
            psi = self.hist[:, :self.Nphi]
        if t is None:
            t = self.hist_t[-len(psi):]

        (psi,), lbl = normalized_y(reference_y, self.state_labels, psi)
        (t,), t_lbl = normalized_time(reference_t, t)
        assert t is not None

        if t_zoom is None:
            t_zoom = int(self.t_CR / self.dt)
        nrows = min(self.Nphi, max_modes)

        fig = plt.figure(figsize=(8, nrows+1), layout="constrained")
        plt.suptitle('State time evolution')
        axs = fig.subplots(nrows, 2, sharey='row', sharex='col')
        if nrows == 1:
            axs = [axs]

        for ii, ax in enumerate(axs):
            ax[0].plot(t, psi[:, ii].real,  label='Real part')
            ax[1].plot(t[-t_zoom:], psi[-t_zoom:, ii].real,  label='Real')
            if np.iscomplexobj(psi[:, ii]):
                ax[0].plot(t, psi[:, ii].imag, label='Imag part')
                ax[1].plot(t[-t_zoom:], psi[-t_zoom:, ii].imag, label='Imag')
                ax[1].legend(fontsize='x-small', ncol=2)
            ax[0].set(ylabel=lbl[ii])
            if ii == nrows-1:
                ax[0].set(xlabel=t_lbl, xlim=[t[0], t[-t_zoom]])
                ax[1].set(xlabel=t_lbl, xlim=[t[-t_zoom], t[-1]])


    def visualize_observable_hist(self, y=None, t=None, t_zoom=None,
                                  reference_y=1.0, reference_t: float = 1.0):
        """Plot the time evolution of each observable.

        Two panels per observable: the full history, and a zoomed-in view of
        the last `t_zoom` steps.

        Parameters
        ----------
        y : ndarray, optional
            Observable history to plot; defaults to `get_observable_hist`.
        t : ndarray, optional
            Matching time stamps; defaults to the tail of `hist_t`.
        t_zoom : int, optional
            Number of trailing steps shown in the zoomed panel; defaults to
            ``t_CR / dt``.
        reference_y : float
            Reference value `y` is divided by (also appended to the axis
            label).
        reference_t : float
            Reference value used to normalise `t` (and its axis label).
        """
        if y is None:
            y = self.get_observable_hist()
        if t is None:
            t = self.hist_t[-len(y):]

        lbl = list(self.obs_labels)
        (t,), t_lbl = normalized_time(reference_t, t)
        assert t is not None
        if reference_y != 1.0:
            y = y / reference_y
            lbl = [f'{lb} / {reference_y}' for lb in lbl]

        if t_zoom is None:
            t_zoom = int(self.t_CR / self.dt)

        fig = plt.figure(figsize=(8, self.Nq+1), layout="constrained")
        plt.suptitle('Observables time evolution')
        axs = fig.subplots(self.Nq, 2, sharey='row', sharex='col')
        if self.Nq == 1:
            axs = [axs]

        for ii, ax in enumerate(axs):
            ax[0].plot(t, y[:, ii])
            ax[1].plot(t[-t_zoom:], y[-t_zoom:, ii])
            ax[0].set(ylabel=lbl[ii])
            if ii == self.Nq-1:
                ax[0].set(xlabel=t_lbl, xlim=[t[0], t[-t_zoom]])
                ax[1].set(xlabel=t_lbl, xlim=[t[-t_zoom], t[-1]])


    def visualize_spatiotemporal_hist(self, reference_y=1.0, reference_t: float = 1.0, **kwargs):
        """No-op hook for subclasses to plot spatiotemporal (e.g. field)
        histories. Called by `visualize_history`."""
        pass






def plot_state_distribution(
    model: Model,
    time_indices=None,
    max_modes=None,
    reference_alpha=None,
    reference_y=1.0,
    reference_t: float = 1.0,
    nbins: int = 6,
) -> None:
    """Histograms of state variables and parameters at given time indices."""
    time_indices = time_indices if time_indices is not None else [-1]
    max_modes    = max_modes if max_modes is not None else model.Nphi

    ncols_phi = min(max_modes, 4)
    nrows_phi = int(np.ceil(max_modes / ncols_phi))

    has_alpha = model.Na > 0
    alpha_hist = alpha_labels = est_alpha = None
    if has_alpha:
        ncols_alpha = min(model.Na, 4)
        nrows_alpha = int(np.ceil(model.Na / ncols_alpha))
        est_alpha   = model.est_alpha
        alpha_hist, alpha_labels = normalized_alpha(
            model.hist[:, -model.Na:, :], est_alpha, model.alpha_labels, reference_a=reference_alpha
        )
    else:
        ncols_alpha = nrows_alpha = 0

    (phi_hist,), state_labels = normalized_y(reference_y, model.state_labels, model.hist[:, :model.Nphi, :])
    (hist_t,), t_lbl          = normalized_time(reference_t, model.hist_t)

    def add_stats_text(ax, yy):
        mean = np.mean(yy)
        std  = np.std(yy) / mean if mean != 0 else 0
        ax.text(0.95, 0.95, f'Mean: {mean:.4f}\nStd: {std:.4f}',
                transform=ax.transAxes, fontsize='x-small',
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    def flat_axs(sf_panel, ncols, nrows):
        axs = sf_panel.subplots(ncols=ncols, nrows=nrows, sharey=True)
        return axs.ravel() if ncols * nrows > 1 else [axs]

    for ti in time_indices:
        phi  = phi_hist[ti, :model.Nphi, :]
        lbls = list(state_labels)

        if np.iscomplexobj(phi):
            phi_c = phi.copy()
            phi   = np.zeros((2 * model.Nphi, phi_c.shape[1]))
            phi[0::2] = phi_c.real
            phi[1::2] = phi_c.imag
            if ti == time_indices[0]:
                lbls      = [f'{lb} ({p})' for lb in state_labels for p in ('real', 'imag')]
                nrows_phi = int(np.ceil((2 * max_modes) / ncols_phi))

        fig = plt.figure(
            figsize=(2 * max(ncols_phi, ncols_alpha or 0), 2 * (nrows_phi + nrows_alpha)),
            layout='constrained',
        )
        plt.suptitle(f'Ensemble distributions at {t_lbl}={hist_t[ti]:.3f}')

        sf = (fig.subfigures(nrows=2, ncols=1, height_ratios=[nrows_phi, nrows_alpha],
                             wspace=0.07, hspace=0.15)
              if has_alpha else [fig.subfigures(nrows=1, ncols=1)])

        axs = flat_axs(sf[0], ncols_phi, nrows_phi)
        for ax, ph, lbl in zip(axs, phi, lbls):
            ax.hist(ph, bins=nbins, color='tab:green')
            ax.set(xlabel=lbl)
            add_stats_text(ax, ph)
        for ax in axs[model.Nphi:]:
            ax.axis('off')

        if has_alpha:
            assert alpha_hist is not None and alpha_labels is not None and est_alpha is not None
            axs2 = flat_axs(sf[1], ncols_alpha, nrows_alpha)
            for ax, a, param in zip(axs2, alpha_hist[ti], est_alpha):
                ax.hist(a, bins=nbins)
                ax.set(xlabel=alpha_labels[param])
                add_stats_text(ax, a)
            for ax in axs2[model.Na:]:
                ax.axis('off')
