

import numpy as np

from .utils import interpolate

__all__ = [
    'Integrator',
    'ConstantIntegrator',
    'DiscreteIntegrator',
    'IVPIntegrator',
    'ivp_forecast_helper',
]

from copy import deepcopy
from functools import partial
from sys import platform
from typing import Any

from scipy.integrate import solve_ivp
from typeguard import typechecked

if platform == "darwin" or platform == "ios":
    import multiprocess as mp
else:
    import multiprocessing as mp




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
    either solved member-by-member in parallel (a multiprocessing pool sized
    to `model.m`, created lazily) or, if ``averaged=True``, solved once for
    the ensemble mean with each member's deviation from the mean left
    unchanged (see `advance_ensemble`).

    Parameters
    ----------
    model_instance : object
        See `Integrator.__init__`.
    method : str
        Integration method forwarded to `scipy.integrate.solve_ivp`
        (default ``'RK45'``).
    """
    def __init__(self, model_instance, method: str = 'RK45'):
        super().__init__(model_instance)
        self.method = method

    @property
    def __pool(self):

        if not hasattr(self, '_pool'):
            self._pool = None

        if self._pool is None and self.model.m > 1:
            # Initialize multiprocessing pool
            N_pools = min(self.model.m, mp.cpu_count())
            print(f'Initializing multiprocessing pool for IVPIntegrator with m={self.model.m} and {N_pools} pools.')
            self._pool = mp.Pool(N_pools)
        return self._pool


    def close(self):
        """Terminate and join the multiprocessing pool, if one was created."""
        if hasattr(self, '_pool') and self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

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
            Accepted for interface compatibility but not used: the governing
            equations are called with ``{**model.alpha0, **model.governing_eqns_params}``
            directly rather than with this argument.

        Returns
        -------
        psi : ndarray, shape ``(Nt, N, 1)``
            Forecasted state, excluding the initial condition.
        t : ndarray, shape ``(Nt,)``
            Time stamps spaced by `model.dt`.
        """
        # print('Using IVPIntegrator advance_single')
        pm = self.model

        t_all = np.round(pm.current_time + np.arange(0, Nt + 1) * pm.dt, pm.precision_t)

        psi0 = pm.current_state
        args = pm.governing_eqns_params

        # --- IVP Logic
        psi = [ivp_forecast_helper(y0=psi0[:, 0],
                                    fun=pm.time_derivative,
                                    t=t_all,
                                    params={**pm.alpha0, **args})]

        try:
            psi = np.array(psi).transpose((1, 2, 0))
        except ValueError as e:
            print(f"Error during final array construction: {e}")
            psi = np.array(psi).T.reshape(-1, psi0.shape[0], pm.m)

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
        """
        pm = self.model

        t_all = np.round(pm.current_time + np.arange(0, Nt + 1) * pm.dt, pm.precision_t)

        psi0 = pm.current_state
        args = pm.governing_eqns_params

        # --- IVP Logic (Similar to previous Model.time_integrate) ---

        if not averaged:
            # Ensemble run (using multiprocessing pool)
            alpha_list = pm.get_alpha()
            forecast_part = partial(ivp_forecast_helper,
                                    fun=pm.time_derivative, t=t_all, method=self.method)

            sol = [self.__pool.apply_async(forecast_part,
                                            kwds={'y0': psi0[:, mi].T, 'params': {**args, **alpha_list[mi]}})
                    for mi in range(pm.m)]

            psi = [s.get() for s in sol]

        else:
            # Averaged forecast
            psi_mean0 = np.mean(psi0, axis=1, keepdims=True)
            psi_deviation = psi0 - psi_mean0
            alpha = pm.get_alpha(psi_mean0)[0]

            psi_mean = ivp_forecast_helper(y0=psi_mean0[:, 0],
                                            fun=pm.time_derivative,
                                            t=t_all,
                                            params={**alpha, **args},
                                            method=self.method)

            psi = [psi_mean + psi_deviation[:, ii] for ii in range(pm.m)]


        # Rearrange dimensions to be Nt+1 x N x m and remove initial condition
        try:
            psi = np.array(psi).transpose((1, 2, 0))
        except ValueError as e:
            print(f"Error during final array construction: {e}")
            psi = np.array(psi).T.reshape(-1, psi0.shape[0], pm.m)

        return psi[1:], t_all[1:]



def ivp_forecast_helper(y0, fun, t, params, method='RK45'):
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

    Returns
    -------
    ndarray, shape ``(len(t), len(y0))``
        Solution evaluated at `t`.
    """
    assert len(t) > 1
    part_fun = partial(fun, **params)

    out = solve_ivp(part_fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method=method)
    return out.y.T
# %%
