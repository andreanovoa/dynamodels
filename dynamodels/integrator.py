

import numpy as np

from .utils import interpolate

__all__ = [
    'Integrator',
    'ConstantIntegrator',
    'DiscreteIntegrator',
    'IVPIntegrator',
    'ivp_forecast_helper',
]

import logging
import math
from copy import deepcopy
from functools import cache, partial
from sys import platform
from typing import Any

from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, OdeSolver, Radau, solve_ivp
from typeguard import typechecked

if platform == "darwin" or platform == "ios":
    import multiprocess as mp
else:
    import multiprocessing as mp

logger = logging.getLogger(__name__)




# %% =================================== INTEGRATOR BASE CLASS ============================================= %% #
class Integrator:
    r"""Abstract base class for the time-integration strategies.

    Defines the interface for advancing the model state; child classes implement
    ``advance_single`` and ``advance_ensemble``. Three strategies are provided:

    - `IVPIntegrator` — continuous, variable-step integration with SciPy's
      ``solve_ivp`` of $\dot{\boldsymbol{\psi}} = f(t, \boldsymbol{\psi},
      \boldsymbol{\alpha})$; the model must define ``time_derivative``.
    - `DiscreteIntegrator` — fixed, discrete-step maps
      $\boldsymbol{\psi}_{t+\Delta t} = F(\boldsymbol{\psi}_t,
      \boldsymbol{\alpha})$ (e.g., ETDRK4, ESN); the model must define
      ``time_step``.
    - `ConstantIntegrator` — holds the state constant,
      $\boldsymbol{\psi}(t) = \boldsymbol{\psi}(0)$.
    """

    @typechecked
    def __init__(self, model_instance: object):
        """Initialize the integrator with a model instance.

        Parameters
        ----------
        model_instance : object
            The (not necessarily `Model`) instance to integrate; must expose
            `time_derivative`/`time_step`, `current_state`, `dt`, and the other
            attributes each concrete strategy relies on.
        """

        self.model = model_instance

    @property
    def is_ensemble(self):
        """bool: True if `model.current_state` carries more than one member
        (its last axis has size $>1$), i.e. `advance` should dispatch to
        `advance_ensemble` rather than `advance_single`.
        """
        current_state = self.model.current_state
        if current_state.ndim >= 2 and current_state.shape[-1] > 1:
            return True
        else:
            return False

    def close(self):
        """ Close resources held by the integrator (e.g., multiprocessing pools). """
        pass

    def advance(self, averaged=False, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Common entry point for all integrators; dispatches to
        `advance_single` or `advance_ensemble` based on `is_ensemble`.

        Parameters
        ----------
        averaged : bool
            Forwarded to `advance_ensemble` when the model carries an
            ensemble; ignored for a single member.
        **kwargs
            Forwarded to `advance_single` / `advance_ensemble` (typically
            `Nt` and `alpha`).

        Returns
        -------
        psi : ndarray
            Forecasted state, excluding the initial condition.
        t : ndarray
            Corresponding time stamps.
        """
        if not self.is_ensemble:
            return self.advance_single(**kwargs)
        else:
            return self.advance_ensemble(averaged=averaged, **kwargs)


    def advance_single(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Advance a single (non-ensemble) member. Must be implemented by
        child classes.

        Returns
        -------
        psi : ndarray
            Forecasted state, excluding the initial condition.
        t : ndarray
            Corresponding time stamps.
        """
        raise NotImplementedError("Child Integrator class must implement the advance_single() method.")


    def advance_ensemble(self, Nt: int = 100, averaged: bool = False, alpha: dict[str, Any] = None) -> tuple[np.ndarray, np.ndarray]:
        """Advance every ensemble member. Must be implemented by child
        classes.

        Parameters
        ----------
        Nt : int
            Number of forecast steps.
        averaged : bool
            Strategy-dependent flag controlling whether members are
            propagated individually or only the ensemble mean is integrated.
        alpha : dict, optional
            Parameter values forwarded to the governing equations.

        Returns
        -------
        psi : ndarray
            Forecasted ensemble state, excluding the initial condition.
        t : ndarray
            Corresponding time stamps.
        """
        raise NotImplementedError("Child Integrator class must implement the advance_ensemble() method.")


class ConstantIntegrator(Integrator):
    r"""Integrator that holds the state constant over time, $\psi(t)=\psi(0)$.

    Useful for testing or as a placeholder while other model components (e.g.
    a bias model) are advanced.
    """

    def __init__(self, model_instance):
        """See `Integrator.__init__`."""
        super().__init__(model_instance)

    def advance_single(self, Nt: int = 100, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Repeat `model.current_state` at every output time.

        Parameters
        ----------
        Nt : int
            Number of forecast steps.
        **kwargs
            Accepted for interface compatibility with `advance_ensemble`
            (e.g. `averaged`, `alpha`) but not used: the state is simply held
            constant regardless of these.

        Returns
        -------
        psi : ndarray, shape ``(Nt, N, m)``
            `current_state` repeated `Nt` times, excluding the initial
            condition.
        t : ndarray, shape ``(Nt,)``
            Time stamps spaced by `model.dt`.
        """
        model = self.model
        t_out = np.round(model.current_time + np.arange(Nt + 1) * model.dt, model.precision_t)
        psi = np.repeat(model.current_state[:, :, np.newaxis], Nt + 1, axis=2)
        # return psi, t_out, psi shoud have dimensions Nt x N x m
        psi = psi.transpose((2, 0, 1))  # Nt+1 x N x m

        return psi[1:], t_out[1:]

    def advance_ensemble(self, Nt: int = 100, averaged: bool = False, alpha: dict[str, Any] = None) -> tuple[np.ndarray, np.ndarray]:
        """Identical to `advance_single` (the state is constant regardless of
        ensemble size); `averaged` and `alpha` are forwarded but unused.
        """
        return self.advance_single(Nt=Nt, averaged=averaged, alpha=alpha)



# %% =================================== CONCRETE STRATEGY 2: DISCRETE STEP ============================================= %% #
class DiscreteIntegrator(Integrator):
    r"""Integrator for models advanced by a fixed, discrete map or scheme
    (single member), $\psi_{t+\mathrm{d}t}=F(\psi_t,\alpha)$ (e.g. ETDRK4 in
    `KS`, or an ESN).

    The model must define ``time_step(Nt)``, stepping at its own internal
    time step `dt_step` (`dt_integrator`); if this differs from the model's
    output step `dt` (`dt_output`), the integrator's output is linearly
    interpolated back onto the requested output times.

    Attributes
    ----------
    dt_output : float
        Output time step, `model.dt`.
    dt_integrator : float
        Time step used internally by `model.time_step`, `model.dt_step`.
    relation_integrator_output : float
        ``dt_output / dt_integrator`` (1.0 if the two coincide); the number
        of internal steps per output step.
    """

    def __init__(self, model_instance):
        """See `Integrator.__init__`; also derives `dt_output`,
        `dt_integrator`, and `relation_integrator_output` from the model.
        """
        super().__init__(model_instance)

        self.dt_output = getattr(self.model, 'dt')
        self.dt_integrator = getattr(self.model, 'dt_step')

        if self.dt_output != self.dt_integrator:
            self.relation_integrator_output = self.dt_output / self.dt_integrator
            self.relation_integrator_output = round(self.relation_integrator_output, self.model.precision_t)
        else:
            self.relation_integrator_output = 1.0


    def advance_single(self, Nt: int = 100, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Advance the model via `model.time_step`, resampled onto the
        requested output times.

        Parameters
        ----------
        Nt : int
            Number of *output* forecast steps (at spacing `dt_output`).
        **kwargs
            Accepted for interface compatibility with `advance_ensemble`
            (e.g. `averaged`, `alpha`) but not used here.

        Returns
        -------
        psi : ndarray, shape ``(Nt, N, m)``
            Forecasted state at the output times, excluding the initial
            condition. Taken directly from `model.time_step` if its internal
            time grid already coincides with the output grid; otherwise
            linearly interpolated onto it (never extrapolated beyond the
            integrator's own time range).
        t : ndarray, shape ``(Nt,)``
            Output time stamps, spaced by `dt_output`.
        """
        model = self.model

        t_out = np.round(model.current_time + np.arange(Nt + 1) * self.dt_output, model.precision_t)

        Nt_step = int(np.ceil(Nt * self.relation_integrator_output))

        psi, t = model.time_step(Nt=Nt_step)

        # Equal lengths do not imply equal times: with dt_output != dt_integrator a short
        # request (e.g. Nt=1 at upsample=2 -> Nt_step=1) gives two arrays of the same
        # length spanning different intervals, and returning the integrator's own times
        # would overshoot t_out[-1]. Only skip the interpolation when the two grids
        # genuinely coincide.
        if self.relation_integrator_output == 1.0 and len(t_out) == len(t):
            return psi[1:], t[1:]
        else:
            # Interpolate
            assert t[-1] >= t_out[-1], f"do not extrapolate beyond the integrator time range, {t[-1]} vs {t_out[-1]}"

            psi_interp = interpolate(t, psi, t_eval=t_out, fill_values='extrapolate')
            # model.reset_last_state(psi_interp[-1], t_out[-1])
            return psi_interp[1:], t_out[1:]

    def advance_ensemble(self, Nt = 100, averaged = False, alpha = None):
        """Identical to `advance_single`; `averaged` and `alpha` are
        forwarded but unused (the discrete map is applied uniformly to
        however many members `model.time_step` handles internally).
        """
        return self.advance_single(Nt, averaged=averaged, alpha=alpha)





# %% ===================================  IVP SOLVER ============================================= %% #
class IVPIntegrator(Integrator):
    r"""Integrator using SciPy's `solve_ivp` for continuous, variable-step
    integration of $\dot{\psi}=f(t,\psi,\alpha)$.

    The model must define ``time_derivative(t, psi, **params)``. A single
    member is solved directly with `ivp_forecast_helper`; an ensemble is
    either solved member-by-member in parallel, by a pool of
    ``min(m, cpu_count)`` worker processes, or, if ``averaged=True``, solved
    once for the ensemble mean with each member's deviation from the mean
    left unchanged (see `advance_ensemble`). A member, alone or in an
    ensemble, is solved with its own parameters (`Model.get_alpha`). The
    first forecast with ``averaged=False`` starts the pool. `close`, a deep
    copy (`Model.copy`), `Model.create_long_timeseries`, `Model.reset_model`
    and `Model.init_ensemble` with a new number of members close it, and the
    next forecast with ``averaged=False`` starts a new pool. A pool serves
    one number of members: if the number of members changes otherwise, e.g.
    with `Model.update_history` and ``reset=True``, the next forecast closes
    the pool and starts a new one. The start and the closing are logged at
    DEBUG, with the model and the integrator (see `dynamodels.set_verbose`).

    Parameters
    ----------
    model_instance : object
        See `Integrator.__init__`.
    method : str
        Integration method forwarded to `scipy.integrate.solve_ivp`
        (default ``'RK45'``), for a single member, for each member of an
        ensemble and for the mean of an averaged ensemble. A `Model` builds
        its integrator with the default, so the method of a model is chosen
        after construction, e.g. ``model.integrator.method = 'DOP853'``.
        The method is an attribute of each integrator, set by the
        constructor: setting it on the class
        (``IVPIntegrator.method = 'DOP853'``) has no effect, unlike
        `max_steps`.

    Attributes
    ----------
    max_steps : int or float
        Maximum number of solver steps within one output step `model.dt`
        (default 10000; ``np.inf`` removes the limit). A member that needs
        more steps, typically a diverged state whose time scale has
        collapsed, stops instead of stalling `solve_ivp`, and the forecast
        raises a `RuntimeError` that names the member. With their default
        `dt`, healthy forecasts of the models in dynamodels, over the
        transient and 2000 output steps on the attractor, take at most 12
        solver steps per output step with RK45, RK23 or DOP853, and at most
        60 with Radau, BDF or LSODA. The most are those of Rijke with LSODA:
        45 with the default parameters, and 58 in the chaotic regime
        (``case='chaotic'``). The number of steps grows with `dt`, so a very
        coarse output step needs a larger `max_steps` too: Lorenz63 takes
        about 12 steps of RK45 per time unit and reaches the default at about
        ``dt = 800``. Set it on one integrator
        (``model.integrator.max_steps = 10**5``) or on the class
        (``IVPIntegrator.max_steps = 10**5``), which changes the default of
        every integrator that has not set its own. Unlike `method`, it is a
        class attribute.
    """
    max_steps = 10_000

    def __init__(self, model_instance, method: str = 'RK45'):
        super().__init__(model_instance)
        self.method = method

    @property
    def __pool(self):

        if not hasattr(self, '_pool'):
            self._pool = None

        if self._pool is not None and getattr(self, '_pool_m', None) != self.model.m:
            # the pool was started for another number of members (e.g. before update_history with reset=True)
            self.close()

        if self._pool is None and self.model.m > 1:
            # One pool of worker processes, shared by the members
            n_workers = min(self.model.m, mp.cpu_count())
            logger.debug('%s: starting a pool of %d worker %s for the %d ensemble members (%s).',
                         _model_name(self.model), n_workers, 'process' if n_workers == 1 else 'processes',
                         self.model.m, type(self).__name__)
            self._pool, self._pool_m = mp.Pool(n_workers), self.model.m
        return self._pool


    def close(self):
        """Terminate and join the multiprocessing pool, if one was started; the next forecast of an ensemble with
        ``averaged=False`` starts a new pool.

        The closing is logged at DEBUG with the model and the integrator; nothing is logged if there is no pool.
        """
        if hasattr(self, '_pool') and self._pool is not None:
            logger.debug('%s: closing the pool of worker processes (%s).', _model_name(self.model), type(self).__name__)
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __getstate__(self):
        # pickling a model mid-run must not require a manual close(): drop the
        # live pool (unpicklable); the next ensemble forecast with averaged=False
        # starts a new one
        state = self.__dict__.copy()
        state['_pool'] = None
        return state

    def __deepcopy__(self, memo):
        # Always close the pool before copying so the copy starts with no pool.
        # Re-creating a Pool inside a process that already owns one can deadlock
        # on macOS (fork-based mp).
        self.close()
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def _check_finite(self, psi0):
        """Raise a ValueError that names the members whose initial state `psi0` (N, m) holds a NaN or an inf.

        `solve_ivp` rejects such a state too, but inside the solve of one member (possibly in a worker of the pool),
        which does not know the index of the member.
        """
        if not np.isfinite(psi0).all():
            bad = np.flatnonzero(~np.isfinite(psi0).all(axis=0))
            raise ValueError(f'{_model_name(self.model)}: the initial state of member(s) {bad.tolist()} is not '
                             'finite (NaN or inf); solve_ivp needs a finite initial state')

    def advance_single(self, Nt = 100, averaged=False, alpha = None):
        """Solve the IVP for the single (non-ensemble) member.

        Parameters
        ----------
        Nt : int
            Number of forecast steps.
        averaged : bool
            Accepted for interface compatibility with `advance_ensemble` but
            not used: with one member there is nothing to average.
        alpha : dict, optional
            Accepted for interface compatibility but not used: as in
            `advance_ensemble`, the governing equations are called with the
            parameters of the member, ``model.get_alpha()[0]``, i.e. `alpha0`
            with the values of the estimated parameters (`est_alpha`) in the
            state of the member, and with `model.governing_eqns_params`. A
            model that has no `get_alpha` (it need not be a `Model`) is
            called with its `alpha0`.

        Returns
        -------
        psi : ndarray, shape ``(Nt, N, 1)``
            Forecasted state, excluding the initial condition.
        t : ndarray, shape ``(Nt,)``
            Time stamps spaced by `model.dt`.

        Raises
        ------
        ValueError
            If the initial state is not finite.
        RuntimeError
            If the integration stops before the end of the forecast (see
            `advance_ensemble`).
        """
        # print('Using IVPIntegrator advance_single')
        pm = self.model

        t_all = np.round(pm.current_time + np.arange(0, Nt + 1) * pm.dt, pm.precision_t)

        psi0 = pm.current_state
        self._check_finite(psi0)
        args = pm.governing_eqns_params
        # with the estimated parameters; a duck-typed model may have alpha0 only, as dynamodels 0.6.0 required
        get_alpha = getattr(pm, 'get_alpha', None)
        alpha = get_alpha()[0] if get_alpha is not None else pm.alpha0

        # --- IVP Logic
        psi, reason = ivp_forecast_helper(y0=psi0[:, 0],
                                          fun=pm.time_derivative,
                                          t=t_all,
                                          params={**alpha, **args},
                                          method=self.method,
                                          max_steps=self.max_steps,
                                          full_output=True)

        psi = _stack_members([psi], t_all, [reason], _model_name(pm), pm.precision_t)
        return psi[1:], t_all[1:]


    def advance_ensemble(self, Nt=100, averaged=False, alpha=None):
        r"""Solve the IVP for every ensemble member.

        Parameters
        ----------
        Nt : int
            Number of forecast steps.
        averaged : bool
            If False (default), each member is solved independently and in
            parallel (via a multiprocessing pool), each with its own
            parameters from `model.get_alpha`. If True, the IVP is solved
            once for the ensemble *mean* $\overline{\psi}_0$, and each
            member's forecast is reconstructed as
            $\overline{\psi}(t) + (\psi_{0,i}-\overline{\psi}_0)$, i.e. its
            initial deviation from the mean is carried forward unchanged
            rather than being independently propagated.
        alpha : dict, optional
            Accepted for interface compatibility but not used: parameters are
            always taken from `model.get_alpha` (or `model.get_alpha` on the
            ensemble mean, if `averaged`).

        Returns
        -------
        psi : ndarray, shape ``(Nt, N, m)``
            Forecasted ensemble state, excluding the initial condition.
        t : ndarray, shape ``(Nt,)``
            Time stamps spaced by `model.dt`.

        Raises
        ------
        ValueError
            If the initial state of a member is not finite. The message names
            the members, and no member is integrated.
        RuntimeError
            If the integration of a member stops before the end of the
            forecast. The message names the model, the member, the last
            output time it reached and the reason: the message of
            `solve_ivp`, a time derivative that is not finite at the initial
            state, a step that meets a NaN or an inf, or more than
            `max_steps` solver steps within one output step. With
            ``averaged=True``, every member stops with the mean.
        """
        pm = self.model

        t_all = np.round(pm.current_time + np.arange(0, Nt + 1) * pm.dt, pm.precision_t)

        psi0 = pm.current_state
        self._check_finite(psi0)       # before the pool starts and before the mean of the averaged forecast
        args = pm.governing_eqns_params

        # --- IVP Logic (Similar to previous Model.time_integrate) ---

        if not averaged:
            # Ensemble run (using multiprocessing pool)
            alpha_list = pm.get_alpha()
            # keyword arguments only (kwds), as in dynamodels 0.6.0: a serial stand-in for the pool, e.g. in the example
            # runners of biasda, calls f(**kwds)
            sol = [self.__pool.apply_async(ivp_forecast_helper,
                                            kwds=dict(y0=psi0[:, mi].T, fun=pm.time_derivative, t=t_all,
                                                      params={**args, **alpha_list[mi]}, method=self.method,
                                                      max_steps=self.max_steps, full_output=True))
                    for mi in range(pm.m)]

            psi, reasons = zip(*[s.get() for s in sol])

        else:
            # Averaged forecast
            psi_mean0 = np.mean(psi0, axis=1, keepdims=True)
            psi_deviation = psi0 - psi_mean0
            alpha = pm.get_alpha(psi_mean0)[0]

            psi_mean, reason = ivp_forecast_helper(y0=psi_mean0[:, 0],
                                                   fun=pm.time_derivative,
                                                   t=t_all,
                                                   params={**alpha, **args},
                                                   method=self.method,
                                                   max_steps=self.max_steps,
                                                   full_output=True)

            psi = [psi_mean + psi_deviation[:, ii] for ii in range(pm.m)]
            reasons = [reason and f'the ensemble mean (averaged=True) stopped: {reason}'] * pm.m


        # Rearrange dimensions to be Nt+1 x N x m and remove initial condition
        psi = _stack_members(psi, t_all, reasons, _model_name(pm), pm.precision_t)
        return psi[1:], t_all[1:]



def _model_name(model):
    """Name of the (duck-typed) model in error messages: `model.name` if it has one, else its class name."""
    return getattr(model, 'name', type(model).__name__)


def _time_str(t, decimals=12):
    """The time `t` in an error message, rounded to `decimals` decimals and without trailing zeros, e.g. '0.3' or
    '123456.78'.

    A fixed number of decimals tells the output times apart at any time; a fixed number of significant digits does
    not (with .6g, 123456.78 and 123456.83 both print as 123457). A negative `decimals`, the `precision_t` of a model
    with ``dt >= 1000``, prints the time without decimals.
    """
    return np.format_float_positional(t, precision=max(0, decimals), trim='-')


def _stack_members(psi, t, reasons=None, name='model', decimals=12):
    """The ensemble forecast (n_t, N, m) from the solutions (n_t, N) of the m members at the n_t output times `t`.

    A member whose integration stopped before ``t[-1]`` has fewer rows, and raises a RuntimeError that names the model
    (`name`), the member, the last output time it reached and the reason, ``reasons[i]`` (None if unknown). The times
    are printed with `decimals` decimals (`IVPIntegrator` passes the `precision_t` of the model).
    """
    short = [i for i, p in enumerate(psi) if len(p) != len(t)]
    if short:
        reasons = reasons or [None] * len(psi)
        groups = {}                                     # members that stopped at the same time for the same reason
        for i in short:
            groups.setdefault((len(psi[i]), reasons[i]), []).append(i)
        lines = [f'  member(s) {idx} reached {f"t = {_time_str(t[n - 1], decimals)}" if n else "no output time"} '
                 f'({n} of {len(t)} output times): {reason or "the state diverged or the integration failed"}'
                 for (n, reason), idx in groups.items()]
        raise RuntimeError(f'{name}: solve_ivp stopped before the end of the forecast (t = {_time_str(t[0], decimals)} '
                           f'to {_time_str(t[-1], decimals)}) for member(s) {short}:\n' + '\n'.join(lines))
    return np.array(psi).transpose((1, 2, 0))


@cache
def _bounded(solver):
    """Subclass of the scipy `OdeSolver` class `solver` that fails with a reason, instead of stalling `solve_ivp` or
    raising an error that does not name the member.

    The subclass fails through the documented failure path of `OdeSolver._step_impl`, so that `solve_ivp` returns
    status -1, the reason, and the solution up to the last output time reached. It fails on

    - more than `max_steps` steps within one output step `dt_out`. `solve_ivp` has no limit on the number of steps,
      and a diverged state can collapse the time scale of the system (Lorenz63 at x = 1e200 needs steps of 1e-201);
    - a time derivative that is not finite at the initial state, which gives the Runge-Kutta methods a NaN first
      step that they reject forever;
    - a NaN or an inf in the LU factorization of BDF and Radau, which raises a ValueError;
    - a step of LSODA to a state that is not finite, which LSODA accepts.
    """
    check_state = issubclass(solver, LSODA)     # the other methods reject a step to a state that is not finite

    class Bounded(solver):
        def __init__(self, fun, t0, y0, t_bound, dt_out=np.inf, max_steps=np.inf, **options):
            super().__init__(fun, t0, y0, t_bound, **options)
            self._dt_out, self._max_steps, self._t_mark, self._n = dt_out, max_steps, t0, 0
            # decimals of the times in the reasons, as Model.dt derives precision_t: enough to place a time between
            # two output times, at any time
            self._decimals = min(12, max(0, math.ceil(2 - math.log10(dt_out)))) if 0 < dt_out < np.inf else 12
            # select_initial_step gives a NaN first step if the time derivative at y0 is NaN, and a zero first step if
            # it is inf (or so large that its norm overflows, which is not a failure); LSODA has no h_abs, BDF no f
            h0 = getattr(self, 'h_abs', 1.)
            self._bad_start = math.isnan(h0) or (h0 == 0 and not np.isfinite(getattr(self, 'f', 0.)).all())

        def _time(self, t):
            """The time `t` in a reason (formatted only on a failure, not at every step)."""
            return _time_str(t, self._decimals)

        def _step_impl(self):
            if self._bad_start:
                return False, f'the time derivative at the initial state (t = {self._time(self.t)}) is not finite'
            if abs(self.t - self._t_mark) >= self._dt_out:     # one output step further: a new budget of steps
                self._t_mark, self._n = self.t, 0
            self._n += 1
            if self._n > self._max_steps:
                h = self.step_size                              # None before the first step
                return False, (f'more than {self._max_steps} steps of {solver.__name__} within one output step '
                               f'({self._dt_out:.3g}) at t = {self._time(self.t)}'
                               + (f', step size {h:.3g}' if h else '')
                               + ': the state may have diverged, or the system is too stiff for this method '
                               '(see IVPIntegrator.max_steps)')
            t = self.t
            try:
                success, message = super()._step_impl()
            except ValueError as e:         # lu_factor (BDF, Radau) rejects a matrix with a NaN or an inf
                if 'infs or NaNs' not in str(e):
                    raise
                return False, (f'the step of {solver.__name__} from t = {self._time(t)} met a NaN or an inf ({e}): '
                               'the state may have diverged')
            if success and check_state and not np.isfinite(self.y).all():
                return False, (f'the step of {solver.__name__} from t = {self._time(t)} gave a state that is not '
                               'finite (NaN or inf): the state may have diverged')
            return success, message

    Bounded.__name__ = Bounded.__qualname__ = f'Bounded{solver.__name__}'
    return Bounded


_SOLVERS = {s.__name__: s for s in (RK23, RK45, DOP853, Radau, BDF, LSODA)}    # the methods of solve_ivp


def ivp_forecast_helper(y0, fun, t, params, method='RK45', max_steps=None, full_output=False):
    """Solve a single IVP with `scipy.integrate.solve_ivp`.

    Defined at module level (rather than as a method) so it can be pickled
    for use with `multiprocessing`.

    Parameters
    ----------
    y0 : ndarray
        Initial condition.
    fun : callable
        Right-hand side ``fun(t, y, **params)`` (i.e. `model.time_derivative`).
    t : ndarray
        Evaluation times; ``solve_ivp`` is called over ``(t[0], t[-1])`` with
        ``t_eval=t``. Must have at least two entries.
    params : dict
        Extra keyword arguments bound to `fun` via `functools.partial`.
    method : str
        Integration method forwarded to `solve_ivp` (default ``'RK45'``).
    max_steps : int or float, optional
        Maximum number of solver steps within one output step
        ``(t[-1] - t[0]) / (len(t) - 1)``, beyond which the integration
        stops. The default, None, sets no limit; `IVPIntegrator` passes its
        `max_steps`.
    full_output : bool
        If True, also return the reason why the integration stopped early.

    Returns
    -------
    y : ndarray, shape ``(n, len(y0))``
        Solution evaluated at the first ``n`` times of `t`. If the
        integration stops early, ``n < len(t)``: the time derivative at `y0`
        is not finite, a step meets a NaN or an inf, `solve_ivp` fails, or
        the solver takes more than `max_steps` steps within one output step.
    reason : str or None
        Only if `full_output`: why the integration stopped early, or None if
        it reached ``t[-1]``.
    """
    assert len(t) > 1
    part_fun = partial(fun, **params)

    solver = _SOLVERS.get(method, method) if isinstance(method, str) else method
    options = {}
    if isinstance(solver, type) and issubclass(solver, OdeSolver):
        solver = _bounded(solver)
        options = {'dt_out': abs(t[-1] - t[0]) / (len(t) - 1), 'max_steps': np.inf if max_steps is None else max_steps}
    # otherwise, an unknown method, for which solve_ivp raises its usual ValueError
    out = solve_ivp(part_fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method=solver, **options)
    y = out.y.T if len(out.t) else np.empty((0, len(y0)))   # out.y is an empty list if the first step failed
    reason = None if out.status == 0 else out.message
    return (y, reason) if full_output else y
# %%
