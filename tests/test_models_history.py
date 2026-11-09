"""Tests for the Model classes, the HistoryTracker and the integrators."""
import re

import numpy as np
import pytest
from scipy.integrate import OdeSolver, solve_ivp

import dynamodels.integrator as integrator
from dynamodels import HistoryTracker
from dynamodels.physical import KS, Annular, Lorenz63, Lorenz96, Rijke, VdP


class TestHistoryTracker:

    def make_tracker(self, N=3, m=2, capacity=10):
        h = HistoryTracker(initial_capacity=capacity)
        h.update_history(np.zeros((1, N, m)), t=np.array([0.]), reset=True)
        return h

    def test_reset(self):
        h = self.make_tracker()
        assert h.hist.shape == (1, 3, 2)
        assert h.current_time == 0.

    def test_append_and_current_state(self):
        h = self.make_tracker()
        new = np.random.default_rng(0).normal(size=(4, 3, 2))
        h.update_history(new, t=np.arange(1, 5, dtype=float))
        assert h.hist.shape == (5, 3, 2)
        np.testing.assert_allclose(h.current_state, new[-1])
        assert h.current_time == 4.

    def test_update_last_state(self):
        h = self.make_tracker()
        h.update_history(np.ones((1, 3, 2)), t=np.array([0.5]), modify_saved_states=True)
        assert h.hist.shape == (1, 3, 2)
        np.testing.assert_allclose(h.current_state, 1.)
        assert h.current_time == 0.5

    def test_capacity_grows(self):
        h = self.make_tracker(capacity=3)
        h.update_history(np.zeros((10, 3, 2)), t=np.arange(1, 11, dtype=float))
        assert h.hist.shape[0] == 11
        assert h.capacity >= 11

    def test_time_state_mismatch_raises(self):
        h = self.make_tracker()
        with pytest.raises(AssertionError):
            h.update_history(np.zeros((2, 3, 2)), t=np.array([1.]))


class TestVdP:

    def test_single_forecast(self):
        model = VdP(dt=1e-4)
        psi, t = model.time_integrate(Nt=100)
        assert psi.shape == (100, model.Nphi, 1)
        assert np.isfinite(psi).all()
        model.update_history(psi, t)
        assert model.current_time == pytest.approx(t[-1])

    def test_get_observables(self):
        model = VdP(dt=1e-4)
        psi, t = model.time_integrate(Nt=10)
        model.update_history(psi, t)
        y = model.get_observables()
        assert y.shape == (model.Nq, 1)
        y_hist = model.get_observable_hist()
        assert y_hist.shape == (11, model.Nq, 1)

    def test_observation_operator_shape(self):
        model = VdP(dt=1e-4)
        assert model.M.shape == (model.Nq, model.N)

    def test_parameter_getters(self):
        model = VdP(dt=1e-4, beta=75.)
        assert model.beta == 75.
        assert model.alpha0['beta'] == 75.
        alphas = model.get_alpha()
        assert alphas[0]['beta'] == 75.


class TestLorenz63:

    def test_forecast_is_chaotic_but_finite(self):
        model = Lorenz63(dt=0.02)
        psi, t = model.time_integrate(Nt=500)
        assert np.isfinite(psi).all()
        assert psi.shape[1] == 3


def test_dt_kept_exact():
    """A step such as 1/51200 (a 51.2 kHz sampling rate) is not rounded short; common steps keep their precision."""
    from dynamodels.physical import Annular
    m = Annular(dt=1 / 51200)
    assert m.dt == 1 / 51200
    t = m.time_integrate(Nt=512)[1]
    assert np.isclose(t[-1], 0.01, rtol=0, atol=1e-12)         # 512 steps from t = 0
    m.close()
    v = VdP(dt=1e-4)
    assert v.dt == 1e-4 and v.precision_t == 6
    v.close()


@pytest.fixture
def solve_ivp_methods(monkeypatch):
    """The scipy method (e.g. 'RK23') of every solve_ivp call that the IVP integrator makes in this process."""
    methods = []

    def spy(*args, **kwargs):
        method = kwargs.get('method')
        if isinstance(method, type) and issubclass(method, OdeSolver):     # a solver class, or a subclass of one
            method = next(c.__name__ for c in method.__mro__ if c.__module__.startswith('scipy.'))
        methods.append(method)
        return solve_ivp(*args, **kwargs)

    monkeypatch.setattr(integrator, 'solve_ivp', spy)
    return methods


@pytest.mark.parametrize('m', [1, 3])
def test_ivp_method_reaches_solve_ivp(solve_ivp_methods, m):
    """IVPIntegrator.method reaches solve_ivp for a single member as for an ensemble.

    The averaged ensemble forecast runs in this process, where the spy sees it; a pool of workers would not.
    """
    model = Lorenz63(dt=0.01)
    if m > 1:
        model.init_ensemble(m=m, std_phi=0.1)
    model.integrator.method = 'RK23'
    solve_ivp_methods.clear()
    model.time_integrate(Nt=10, averaged=True)
    model.close()
    assert solve_ivp_methods == ['RK23']


def test_ivp_single_member_forecast_follows_method():
    """A single-member forecast with DOP853 equals a direct DOP853 solve, and an unknown method raises."""
    model = Lorenz63(dt=0.01)
    model.integrator.method = 'DOP853'
    psi0, t0 = model.current_state[:, 0], model.current_time
    psi, t = model.time_integrate(Nt=200)
    params = {**model.alpha0, **model.governing_eqns_params}
    t_all = np.r_[t0, t]
    ref = solve_ivp(lambda tt, y: model.time_derivative(tt, y, **params), (t_all[0], t_all[-1]), psi0,
                    t_eval=t_all, method='DOP853').y.T[1:]
    np.testing.assert_allclose(psi[..., 0], ref, rtol=0, atol=1e-12)     # RK45 differs by about 0.3
    model.integrator.method = 'not-a-method'
    with pytest.raises(ValueError, match='method'):
        model.time_integrate(Nt=5)
    model.close()


def member_forecasts(column, method='RK45', est_alpha=()):
    """The forecast of the member `column` (state and estimated parameters), solved with `method` alone (a single
    member) and as member 0 of an ensemble of two members, which the pool solves (averaged=False)."""
    other = column + np.eye(len(column))[0]                                 # another member, with x larger by 1
    forecasts = []
    for psi0 in (column[:, None], np.stack([column, other], axis=1)):
        model = Lorenz63(dt=0.01)
        model.init_ensemble(m=psi0.shape[1], est_alpha=list(est_alpha), ensemble_psi0=psi0)
        model.integrator.method = method
        forecasts.append(model.time_integrate(Nt=200)[0][..., 0])
        model.close()
    return forecasts


def test_single_member_forecasts_with_its_estimated_parameters():
    """A single member with an estimated parameter forecasts with the value in its state (rho = 40), not with alpha0
    (rho = 28), as the same member does in the pool of an ensemble."""
    alone, pooled = member_forecasts(np.array([1., 2., 3., 40.]), est_alpha=['rho'])
    assert np.array_equal(alone, pooled)                                    # alpha0 differs by about 18
    assert np.all(alone[:, 3] == 40.)                                       # the parameter row is kept


def test_pool_member_follows_method():
    """IVPIntegrator.method reaches the members that the pool solves (averaged=False): a member forecast through the
    pool equals the same member forecast alone with the same method, and DOP853 differs from RK45."""
    alone, pooled = member_forecasts(np.array([1., 2., 3.]), method='DOP853')
    assert np.array_equal(alone, pooled)
    rk45 = Lorenz63(dt=0.01, psi0=[1., 2., 3.]).time_integrate(Nt=200)[0][..., 0]     # the default method, alone
    assert np.abs(rk45 - pooled).max() > 1e-3


def test_method_is_an_attribute_of_each_integrator(monkeypatch):
    """As the docs say: IVPIntegrator.method on the class has no effect, while IVPIntegrator.max_steps sets the default
    of every integrator."""
    monkeypatch.setattr(integrator.IVPIntegrator, 'method', 'DOP853', raising=False)
    monkeypatch.setattr(integrator.IVPIntegrator, 'max_steps', 7)
    model = Lorenz63()
    assert model.integrator.method == 'RK45' and model.integrator.max_steps == 7
    model.close()


@pytest.mark.parametrize('make, kwargs, unknown', [
    (lambda: KS(Nx=32, L=22., dt=0.1), dict(std_alpha={'nu': (0.9, 1.1)}), "['nu']"),  # structural; KS.params is []
    (lambda: Lorenz96(Nx=10), dict(est_alpha=['F', 'Nx']), "['Nx']"),  # structural
    (Lorenz63, dict(est_alpha=['rho', 'rhoo']), "['rhoo']"),  # not an attribute
    (Lorenz63, dict(est_alpha=['copy']), "['copy']"),  # a method
    (VdP, dict(std_alpha={'kappa': (1., 2.), 'foo': (1., 2.)}), "['foo']"),
    (Rijke, dict(est_alpha=['tau', 'foo']), "['foo']"),  # after modify_settings has lengthened the delay line
    # attributes of dynamodels that are not parameters: accepted before, they failed at the forecast
    (lambda: KS(Nx=32, L=22., dt=0.1), dict(est_alpha=['dt']), "['dt']"),  # a property of Model
    (lambda: KS(Nx=32, L=22., dt=0.1), dict(est_alpha=['initial_amplitude', 'Nq']), "['initial_amplitude', 'Nq']"),
    (Rijke, dict(est_alpha=['beta', 'xf']), "['xf']"),  # the flame location, a fixed parameter
    (Rijke, dict(std_alpha={'Nq': (5, 7)}), "['Nq']"),
    (Annular, dict(est_alpha=['ER']), "['ER']"),
    (Lorenz63, dict(est_alpha=['t_lyap', 'seed', 'initialized']), "['t_lyap', 'seed', 'initialized']"),
    (Lorenz63, dict(est_alpha=['Nphi', 'precision_t']), "['Nphi', 'precision_t']"),  # a property, an instance attribute
    (Lorenz63, dict(est_alpha=['_psi0']), "['_psi0']"),  # private
    (lambda: Shadow(), dict(est_alpha=['t_lyap']), "['t_lyap']"),  # a subclass outside dynamodels overrides it
], ids=['KS-nu', 'Lorenz96-Nx', 'Lorenz63-rhoo', 'Lorenz63-copy', 'VdP-foo', 'Rijke-tau-foo', 'KS-dt', 'KS-attributes',
        'Rijke-xf', 'Rijke-Nq', 'Annular-ER', 'Lorenz63-attributes', 'Lorenz63-derived', 'Lorenz63-private', 'Shadow'])
def test_init_ensemble_rejects_a_name_that_cannot_be_estimated(make, kwargs, unknown):
    """A name to estimate that the model does not have, a method, a structural parameter or another attribute of
    dynamodels raises a ValueError at init_ensemble, not an opaque error at the forecast, and leaves the model as it
    was."""
    model = make()
    model.t_transient = 0.
    lims = dict(model.alpha_lims)  # before the copy of the attributes: the first access sets _alpha_lims
    before, history, hist = dict(vars(model)), dict(vars(model.history)), model.hist.copy()
    with pytest.raises(ValueError, match=f'cannot estimate {re.escape(unknown)}'):
        model.init_ensemble(m=3, **kwargs)
    assert vars(model).keys() == before.keys() and all(vars(model)[key] is val for key, val in before.items())
    assert all(vars(model.history)[key] is val for key, val in history.items()) and np.array_equal(model.hist, hist)
    assert model.ensemble_cfg is False and model.alpha_lims == lims
    if isinstance(model, Rijke):
        assert (model.Nc, model.Nphi, model.tau_adv, model.Dc.shape) == (10, 30, 1.5e-3, (11, 11))
    psi = model.time_integrate(Nt=2)[0]  # still forecasts its single member
    assert psi.shape[-1] == 1 and np.all(np.isfinite(psi))
    model.close()


class Shadow(Lorenz63):
    """Overrides an attribute that dynamodels defines, which stays an attribute of dynamodels, not a parameter."""
    t_lyap = 2.


class Registering(Lorenz63):
    """Registers the parameter 'gain' in modify_settings, as the ESN of romda registers the singular values of its
    read-out."""
    def modify_settings(self):
        if 'gain' in self.est_alpha and 'gain' not in self.params:
            self.gain, self.params = 1., [*self.params, 'gain']


class Hyper(Lorenz63):
    """Has an attribute 'gain' outside params, as the ESN of romda has its spectral radius, which it estimates."""
    gain = 1.


@pytest.mark.parametrize('cls', [Registering, Hyper])
def test_init_ensemble_accepts_a_name_that_a_subclass_estimates(cls):
    """A subclass may estimate a parameter that modify_settings registers (the names are checked after it), or a
    class attribute of its own, outside params and outside dynamodels, that its forecast reads for each member."""
    model = cls()
    model.t_transient = 0.
    model.init_ensemble(m=3, est_alpha=['gain'], std_alpha=0.1)
    assert model.est_alpha == ['gain'] and model.Na == 1 and model.hist.shape == (1, 4, 3)
    assert [alpha['gain'] for alpha in model.get_alpha()] == model.current_state[3].tolist()
    model.close()


def _diverging(model):
    model.integrator.max_steps = 0     # the transient forecast of init_ensemble stops at its first solver step


@pytest.mark.parametrize('kwargs, prepare, error', [
    (dict(std_alpha={'rho': 0.1}), None, TypeError),     # a uniform distribution needs (low, high)
    (dict(std_alpha={'rho': (27., 29.)}), _diverging, RuntimeError),
], ids=['std_alpha-format', 'transient-diverges'])
@pytest.mark.parametrize('m0', [1, 2])
def test_init_ensemble_that_fails_later_leaves_the_model_as_it_was(kwargs, prepare, error, m0):
    model = Lorenz63()
    model.t_transient = 0.1
    if m0 > 1:
        model.init_ensemble(m=m0, std_alpha={'beta': (2., 3.)})
    if prepare:
        prepare(model)
    lims, labels = dict(model.alpha_lims), dict(model.alpha_labels)
    before, history, hist = dict(vars(model)), dict(vars(model.history)), model.hist.copy()
    cfg = model.ensemble_cfg if m0 == 1 else dict(model.ensemble_cfg)
    with pytest.raises(error):
        model.init_ensemble(m=3, **kwargs)
    assert vars(model).keys() == before.keys() and all(vars(model)[key] is val for key, val in before.items())
    assert all(vars(model.history)[key] is val for key, val in history.items()) and np.array_equal(model.hist, hist)
    assert model.ensemble_cfg == cfg and (model.alpha_lims, model.alpha_labels) == (lims, labels)
    model.integrator.max_steps = 10_000
    assert np.isfinite(model.time_integrate(Nt=2)[0]).all()
    model.close()


class Relabelling(Lorenz63):
    """Changes the limits and the labels through the public setters in modify_settings, as the ESN of romda does."""

    def modify_settings(self):
        self.alpha_lims = dict(rho=(0., 1.))
        self.alpha_labels = dict(rho='r')


def test_init_ensemble_that_fails_restores_the_limits_and_labels_set_by_modify_settings():
    """The setters of alpha_lims and alpha_labels assign new dicts, so that the undo of a failed init_ensemble restores
    the previous ones."""
    model = Relabelling()
    lims, labels = model.alpha_lims, model.alpha_labels
    saved = dict(lims), dict(labels)
    with pytest.raises(ValueError, match="cannot estimate"):
        model.init_ensemble(m=3, est_alpha=['rho', 'foo'])
    assert model.alpha_lims is lims and model.alpha_labels is labels and (dict(lims), dict(labels)) == saved
    model.t_transient = 0.
    model.init_ensemble(m=3, est_alpha=['rho'])
    assert model.alpha_lims['rho'] == (0., 1.) and model.alpha_labels['rho'] == 'r' and lims['rho'] == saved[0]['rho']
    model.close()


class Spy(Lorenz63):
    """Records the parameters of the members in each forecast."""

    def time_integrate(self, Nt=100, averaged=False):
        self.seen = self.get_alpha()
        return super().time_integrate(Nt=Nt, averaged=averaged)


@pytest.mark.parametrize('m0', [1, 2])
def test_init_ensemble_again_reads_the_parameter_rows_by_their_own_names(m0):
    """In the transient of a second init_ensemble, the members keep the parameters of the first ensemble, which the
    rows of the state hold by the names of the first est_alpha; the new ensemble starts from the state rows only."""
    model = Spy()
    model.t_transient = 0.1
    model.init_ensemble(m=m0, std_alpha={'rho': (30., 32.)})
    rho = model.current_state[3].copy()
    model.init_ensemble(m=3, std_alpha={'beta': (2., 3.)})
    assert [alpha['rho'] for alpha in model.seen] == rho.tolist()
    assert all(alpha['beta'] == Lorenz63.beta for alpha in model.seen)
    assert model.est_alpha == ['beta'] and model.hist.shape == (1, 4, 3)
    assert np.all((2. <= model.current_state[3]) & (model.current_state[3] <= 3.))
    model.init_ensemble(m=4, std_alpha={'beta': (2., 3.)})     # the same names again
    assert model.hist.shape == (1, 4, 4)
    model.close()
