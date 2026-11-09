"""A diverged or non-finite IVP member stops with an error that names it, instead of stalling solve_ivp."""
import re
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dynamodels import Model
from dynamodels.integrator import IVPIntegrator, _stack_members, ivp_forecast_helper
from dynamodels.physical import Lorenz63

GUARD = 10**5       # evaluations of the time derivative after which a test fails instead of hanging

# select_initial_step of scipy divides 0 by 0 when the norm of the time derivative overflows (state at 1e200), and
# the step-size control of BDF and Radau divides by zero or overflows there
scipy_initial_step = pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')
scipy_implicit = pytest.mark.filterwarnings('ignore:divide by zero encountered:RuntimeWarning',
                                            'ignore:overflow encountered:RuntimeWarning')


class Guarded(Lorenz63):
    """Lorenz63 that fails after GUARD evaluations of its time derivative (in each process). The time derivative is
    NaN for 50 < x < 100 and inf for 100 <= x < 1e100, as in a model that overflows."""
    calls = 0

    @staticmethod
    def time_derivative(t, psi, **params):
        Guarded.calls += 1
        if Guarded.calls > GUARD:
            raise AssertionError('the integration did not stop')
        d = np.asarray(Lorenz63.time_derivative(t, psi, **params), dtype=float)
        if 50 < psi[0] < 100:
            return np.full_like(d, np.nan)
        if 100 <= psi[0] < 1e100:
            return np.full_like(d, np.inf)
        return d


class Blowup(Model):
    """dx/dt = a x^2, which blows up at t = 1 / (a x0)."""
    params = ['a']
    a = 1.
    Nq = 1

    @property
    def obs_labels(self):
        return ['$x$']

    @staticmethod
    def time_derivative(t, psi, a):
        return a * psi ** 2


@pytest.fixture(autouse=True)
def reset_guard():
    Guarded.calls = 0       # before any pool of the test is forked


def ensemble(model, column):
    """`model` with an ensemble of 4 members, of which member 2 starts at `column`."""
    psi0 = np.repeat(np.array([[1.], [2.], [3.]]), 4, axis=1)
    psi0[:, 2] = column
    model.init_ensemble(m=4, ensemble_psi0=psi0)
    return model


def test_stack_members_names_model_time_and_reason():
    t = np.arange(6) * 0.1
    full, short, empty = np.zeros((6, 3)), np.zeros((4, 3)), np.zeros((0, 3))
    assert _stack_members([full, full], t).shape == (6, 3, 2)
    with pytest.raises(RuntimeError, match=r'(?s)^Toy: .*\(t = 0 to 0\.5\) for member\(s\) \[1, 2, 3\]:\n'
                                           r'  member\(s\) \[1, 3\] reached t = 0\.3 \(4 of 6 output times\): why\n'
                                           r'  member\(s\) \[2\] reached no output time \(0 of 6 output times\): '
                                           r'the state diverged or the integration failed$'):
        _stack_members([full, short, empty, short], t, [None, 'why', None, 'why'], 'Toy')


@pytest.mark.parametrize('t0, dt, start, end, reached', [(123456.78, 0.01, '123456.78', '123456.83', '123456.82'),
                                                         (1e4, 0.02, '10000', '10000.1', '10000.08')])
def test_stack_members_tells_late_times_apart(t0, dt, start, end, reached):
    """The times are printed with the decimals of the output times: with .6g, the times of the first case all print
    as 123457, and 10000.08 prints as 10000.1, the requested end."""
    t = np.round(t0 + np.arange(6) * dt, 4)
    with pytest.raises(RuntimeError, match=rf'\(t = {re.escape(start)} to {re.escape(end)}\) .*\n  member\(s\) \[1\] '
                                           rf'reached t = {re.escape(reached)} \(5 of 6 output times\): why$'):
        _stack_members([np.zeros((6, 3)), np.zeros((5, 3))], t, [None, 'why'], 'Toy', 4)


def test_stack_members_prints_coarse_times_without_decimals():
    """A model with dt >= 1000 has a negative precision_t, which prints the times without decimals."""
    t = np.round(np.arange(4) * 1000., -1)
    with pytest.raises(RuntimeError, match=r'\(t = 0 to 3000\) .*\n  member\(s\) \[1\] reached t = 1000 \(2 of 4 '
                                           r'output times\): why$'):
        _stack_members([np.zeros((4, 3)), np.zeros((2, 3))], t, [None, 'why'], 'Toy', -1)


@scipy_initial_step
@pytest.mark.parametrize('m', [1, 2], ids=['single', 'pool'])
@pytest.mark.parametrize('dt', [1000., 1e4])
def test_coarse_output_step_names_the_member(m, dt):
    """With dt >= 1000, precision_t < 0: the error names the member instead of failing on the format of the times."""
    model = Lorenz63(dt=dt)
    assert model.precision_t < 0
    if m > 1:
        model.init_ensemble(m=m, ensemble_psi0=np.repeat(model.current_state, m, axis=1))
    with pytest.raises(RuntimeError, match=rf'(?s)^Lorenz63: .*\(t = 0 to {2 * dt:.0f}\) .*reached t = 0 \(1 of 3 '
                                           r'output times\): more than 10000 steps of RK45 .*IVPIntegrator\.max_steps'):
        model.time_integrate(Nt=2)
    model.close()
    blowup = Blowup(psi0=np.array([2.]), dt=dt)
    with pytest.raises(RuntimeError, match=rf'(?s)^Blowup: .*\(t = 0 to {2 * dt:.0f}\) .*reached t = 0 \(1 of 3 '
                                           r'output times\): Required step size'):
        blowup.time_integrate(Nt=2)


class _SerialPool:
    """Serial stand-in for the pool, as in the example runners of biasda: it takes the keyword arguments of a task
    only, as in dynamodels 0.6.0."""

    def apply_async(self, f, kwds):
        out = f(**kwds)
        return SimpleNamespace(get=lambda: out)


def test_pool_tasks_take_keyword_arguments_only(monkeypatch):
    model = Lorenz63()
    model.init_ensemble(m=3, std_alpha={'rho': (27., 29.)})
    reference = model.copy()
    ref, _ = reference.time_integrate(Nt=20)      # with the pool
    reference.close()
    monkeypatch.setattr(IVPIntegrator, '_IVPIntegrator__pool', property(lambda self: _SerialPool()))
    psi, _ = model.time_integrate(Nt=20)
    assert np.array_equal(psi, ref)


def at_time(model, t0):
    """`model`, with its history restarted at the time `t0` from its current state."""
    model.update_history(model.current_state[None], t=np.array([t0]), reset=True)
    return model


@scipy_initial_step
def test_late_forecast_tells_times_apart():
    """A forecast from t = 123456.78 prints the output times and the time of the reason with the decimals of dt."""
    with pytest.raises(RuntimeError, match=r'(?s)^Blowup: .*\(t = 123456\.78 to 123457\.78\) .*reached t = 123457\.27 '
                                           r'\(50 of 101 output times\): Required step size'):
        at_time(Blowup(psi0=np.array([2.]), dt=0.01), 123456.78).time_integrate(Nt=100)
    with pytest.raises(RuntimeError, match=r'(?s)\(t = 123456\.78 to 123456\.88\) .*: the time derivative at the '
                                           r'initial state \(t = 123456\.78\) is not finite$'):
        at_time(Guarded(psi0=np.array([60., 1., 1.])), 123456.78).time_integrate(Nt=5)


@pytest.mark.parametrize('averaged', [False, True])
def test_nonfinite_initial_member_is_named(averaged):
    model = ensemble(Lorenz63(), [np.nan, 1., 1.])
    with pytest.raises(ValueError, match=r'^Lorenz63: the initial state of member\(s\) \[2\] is not finite'):
        model.time_integrate(Nt=5, averaged=averaged)
    assert getattr(model.integrator, '_pool', None) is None         # raised before the pool starts
    model.close()
    with pytest.raises(ValueError, match=r'member\(s\) \[0\] is not finite'):
        Lorenz63(psi0=np.array([1., np.inf, 1.])).time_integrate(Nt=5)


@scipy_initial_step
@pytest.mark.parametrize('m, averaged, members, prefix', [(1, False, r'\[0\]', ''),
                                                          (4, False, r'\[2\]', ''),
                                                          (4, True, r'\[0, 1, 2, 3\]',
                                                           r'the ensemble mean \(averaged=True\) stopped: ')],
                         ids=['single', 'pool', 'averaged'])
def test_huge_member_stops_instead_of_hanging(m, averaged, members, prefix):
    model = Guarded(psi0=np.array([1e200, 1., 1.])) if m == 1 else ensemble(Guarded(), [1e200, 1., 1.])
    model.integrator.max_steps = 1000
    with pytest.raises(RuntimeError, match=rf'^Guarded: .* for member\(s\) {members}:\n  member\(s\) {members} '
                                           rf'reached t = 0 \(1 of 6 output times\): {prefix}more than 1000 steps '
                                           r'of RK45 within one output step \(0\.02\)'):
        model.time_integrate(Nt=5, averaged=averaged)
    model.close()


@scipy_initial_step
@pytest.mark.parametrize('x0', [60., 200.], ids=['nan', 'inf'])
@pytest.mark.parametrize('m', [1, 4])
def test_nonfinite_derivative_at_the_initial_state_stops(m, x0):
    model = Guarded(psi0=np.array([x0, 1., 1.])) if m == 1 else ensemble(Guarded(), [x0, 1., 1.])
    with pytest.raises(RuntimeError, match=rf'member\(s\) \[{0 if m == 1 else 2}\] reached no output time .*: '
                                           r'the time derivative at the initial state \(t = 0\) is not finite'):
        model.time_integrate(Nt=5)
    model.close()


# Why each method of solve_ivp stops from x = 1e200 with max_steps = 100. DOP853 fails at its first step with the
# message of solve_ivp; the LU factorization of BDF and Radau meets a NaN or an inf
HUGE = {'RK23': 'more than 100 steps of RK23 ', 'RK45': 'more than 100 steps of RK45 ',
        'DOP853': 'Required step size', 'Radau': 'the step of Radau from t = 0 met a NaN or an inf',
        'BDF': 'the step of BDF from t = 0 met a NaN or an inf', 'LSODA': 'more than 100 steps of LSODA '}


@scipy_initial_step
@scipy_implicit
@pytest.mark.parametrize('method', list(HUGE))
def test_helper_stops_every_method(method):
    model = Lorenz63()
    t = np.round(np.arange(6) * model.dt, model.precision_t)
    params = {**model.alpha0, **model.governing_eqns_params}
    y, reason = ivp_forecast_helper(np.array([1e200, 1., 1.]), Guarded.time_derivative, t, params, method=method,
                                    max_steps=100, full_output=True)
    assert len(y) < len(t) and reason.startswith(HUGE[method])
    y, reason = ivp_forecast_helper(np.array([60., 1., 1.]), Guarded.time_derivative, t, params, method=method,
                                    full_output=True)
    assert y.shape == (0, 3)
    if method == 'LSODA':       # LSODA has no initial step to check, and its first step gives a NaN state
        assert reason == 'the step of LSODA from t = 0 gave a state that is not finite (NaN or inf): the state may ' \
                         'have diverged'
    else:
        assert reason == 'the time derivative at the initial state (t = 0) is not finite'
    with pytest.raises(ValueError, match='`method` must be one of'):
        ivp_forecast_helper(np.ones(3), Guarded.time_derivative, t, params, method='Euler')


class LateNaN(Lorenz63):
    """Lorenz63 whose time derivative is NaN (or raises a ValueError, with `fail`) after t = 0.05."""
    fail = None

    @staticmethod
    def time_derivative(t, psi, **params):
        if t > 0.05:
            if LateNaN.fail:
                raise ValueError(LateNaN.fail)
            return np.full(3, np.nan)
        return Lorenz63.time_derivative(t, psi, **params)


@pytest.mark.parametrize('method', list(HUGE))
def test_nan_mid_forecast_stops_every_method(method):
    """LSODA accepts a step to a NaN state, and BDF fails in the LU factorization: both stop, as the others do."""
    model = LateNaN()
    model.integrator.method = method
    with pytest.raises(RuntimeError, match=r'^LateNaN: .*\(t = 0 to 0\.2\) for member\(s\) \[0\]:\n  member\(s\) \[0\] '
                                           r'reached t = 0\.04 \(3 of 11 output times\)'):
        model.time_integrate(Nt=10)


def test_other_value_errors_of_the_time_derivative_propagate(monkeypatch):
    monkeypatch.setattr(LateNaN, 'fail', 'a bug in the model')
    model = LateNaN()
    model.integrator.method = 'BDF'
    with pytest.raises(ValueError, match='^a bug in the model$'):
        model.time_integrate(Nt=10)


@scipy_initial_step
@scipy_implicit
@pytest.mark.parametrize('method', ['Radau', 'BDF', 'LSODA'])
@pytest.mark.parametrize('m', [1, 4])
def test_implicit_methods_name_the_diverged_member(m, method):
    """The issue's example, Lorenz63 at x = 1e200, with the implicit methods of solve_ivp."""
    model = Guarded(psi0=np.array([1e200, 1., 1.])) if m == 1 else ensemble(Guarded(), [1e200, 1., 1.])
    model.integrator.method, model.integrator.max_steps = method, 1000
    member = 0 if m == 1 else 2
    reason = 'more than 1000 steps of LSODA ' if method == 'LSODA' else f'the step of {method} from t = 0 met a NaN'
    with pytest.raises(RuntimeError, match=rf'^Guarded: .* for member\(s\) \[{member}\]:\n  member\(s\) \[{member}\] '
                                           rf'reached .*: {reason}'):
        model.time_integrate(Nt=5)
    model.close()


def test_helper_has_no_step_limit_by_default(monkeypatch):
    """Only IVPIntegrator passes its limit: a direct call over a long output step (100 time units) is not cut short."""
    monkeypatch.setattr(IVPIntegrator, 'max_steps', 100)
    model = Lorenz63()
    params = {**model.alpha0, **model.governing_eqns_params}
    y = ivp_forecast_helper(np.ones(3), Lorenz63.time_derivative, np.array([0., 100.]), params)
    assert y.shape == (2, 3) and np.isfinite(y).all()
    y, reason = ivp_forecast_helper(np.ones(3), Lorenz63.time_derivative, np.array([0., 100.]), params,
                                    max_steps=100, full_output=True)
    assert y.shape == (1, 3) and reason.startswith('more than 100 steps of RK45 within one output step (100)')


def test_blowup_reports_time_reached_and_scipy_reason():
    model = Blowup(psi0=np.array([2.]), dt=0.01)
    with pytest.raises(RuntimeError, match=r'(?s)^Blowup: .*\(t = 0 to 1\).*reached t = 0\.49 \(50 of 101 output '
                                           r'times\): Required step size'):
        model.time_integrate(Nt=100)
    model = Blowup(psi0=np.array([0.1]), dt=0.01)
    model.init_ensemble(m=4, ensemble_psi0=np.array([[0.1, 0.2, 2., 4.]]))
    with pytest.raises(RuntimeError, match=r'(?s)for member\(s\) \[2, 3\]:\n  member\(s\) \[2\] reached t = 0\.49 .*\n'
                                           r'  member\(s\) \[3\] reached t = 0\.24 \(25 of 101'):
        model.time_integrate(Nt=100)
    model.close()


def test_healthy_forecast_is_unchanged():
    model = Lorenz63()
    params = {**model.alpha0, **model.governing_eqns_params}
    t = np.round(np.arange(201) * model.dt, model.precision_t)
    y0 = model.current_state[:, 0]
    ref = solve_ivp(lambda tt, y: Lorenz63.time_derivative(tt, y, **params), (t[0], t[-1]), y0, t_eval=t).y.T
    y = ivp_forecast_helper(y0, Lorenz63.time_derivative, t, params)              # the array, as before
    assert np.array_equal(y, ref)
    y, reason = ivp_forecast_helper(y0, Lorenz63.time_derivative, t, params, full_output=True)
    assert reason is None and np.array_equal(y, ref)
    psi, _ = model.time_integrate(Nt=200)
    assert np.array_equal(psi[:, :, 0], ref[1:])


def test_max_steps_can_be_changed(monkeypatch):
    model = Lorenz63(psi0=np.array([1e3, 1., 1.]))
    model.integrator.max_steps = 2
    with pytest.raises(RuntimeError, match='more than 2 steps'):
        model.time_integrate(Nt=2)
    model.integrator.max_steps = np.inf
    assert np.isfinite(model.time_integrate(Nt=2)[0]).all()
    monkeypatch.setattr(IVPIntegrator, 'max_steps', 3)                            # the default of every integrator
    with pytest.raises(RuntimeError, match='more than 3 steps'):
        Lorenz63(psi0=np.array([1e3, 1., 1.])).time_integrate(Nt=2)
