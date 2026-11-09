"""Model.reset_model re-initializes a model in place: the history restarts at t = 0 from psi0, and dt, the parameters,
the integrator and the ensemble configuration are kept unless overridden."""
import warnings
from copy import deepcopy

import numpy as np
import pytest
from typeguard import TypeCheckError

from dynamodels.physical import KS, KS2D, Lorenz63, Lorenz96, Rijke, VdP


def advanced(model, Nt=10):
    """`model`, integrated `Nt` steps further, with the forecast appended to its history."""
    psi, t = model.time_integrate(Nt=Nt)
    model.update_history(psi, t)
    return model


def restarted(model):
    """A copy of `model` whose history restarts at t = 0 from its current state."""
    ref = deepcopy(model)
    ref.update_history(ref.current_state, t=np.array([0.]), reset=True)
    return ref


def test_reset_single_keeps_state_dt_and_parameters():
    m = advanced(Lorenz63(rho=30., dt=0.01))
    x = m.current_state.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # no warning that psi0 is re-assigned
        m.reset_model()
    assert m.hist.shape == (1, 3, 1) and m.hist_t.tolist() == [0.]
    assert np.array_equal(m.current_state, x) and np.array_equal(m.psi0, x)
    assert m.dt == 0.01 and m.alpha0['rho'] == 30.
    ref = Lorenz63(rho=30., dt=0.01, psi0=x[:, 0])
    assert np.array_equal(m.time_integrate(Nt=10)[0], ref.time_integrate(Nt=10)[0])
    m.close(), ref.close()


def test_reset_single_overrides():
    m = advanced(Lorenz63())
    _ = m.filename  # cache the name, as a saved dataset would
    m.reset_model(psi0=[1., 2., 3.], dt=0.01, rho=35.)
    ref = Lorenz63(psi0=[1., 2., 3.], dt=0.01, rho=35.)
    assert m.dt == 0.01 and m.rho == 35. and m.alpha0 == ref.alpha0 == m.alpha
    assert m.filename == ref.filename and m.t_lyap == ref.t_lyap
    a, ta = m.time_integrate(Nt=10)
    b, tb = ref.time_integrate(Nt=10)
    assert np.array_equal(a, b) and np.array_equal(ta, tb)
    m.close(), ref.close()


@pytest.mark.parametrize('cls, kwargs, override', [(Lorenz63, {}, dict(rho=35.)),
                                                   (Lorenz96, dict(Nx=10), dict(F=6.)),
                                                   (Rijke, {}, dict(beta=12.))],
                         ids=['Lorenz63', 'Lorenz96', 'Rijke'])
def test_reset_derives_t_lyap_again(cls, kwargs, override):
    m = cls(**kwargs)
    t_lyap = m.t_lyap
    m.reset_model(**override)
    ref = cls(**kwargs, **override)
    assert m.t_lyap == ref.t_lyap != t_lyap
    m.close(), ref.close()


@pytest.mark.parametrize('est_alpha', [[], ['rho']], ids=['state', 'state_and_rho'])
def test_reset_ensemble(est_alpha):
    m = Lorenz63()
    m.init_ensemble(m=3, std_phi=0.1, est_alpha=est_alpha, std_alpha=0.1)
    advanced(m, Nt=5)
    ref = restarted(m)  # the deep copy closes the pool of m
    m.time_integrate(Nt=1)  # starts it again
    old = m.integrator
    assert old._pool is not None
    filename = m.filename
    m.reset_model()
    assert old._pool is None and m.integrator is not old  # the old pool is released
    assert (m.m, m.Nphi, m.Na, m.est_alpha, m.filename) == (3, 3, len(est_alpha), est_alpha, filename)
    assert m.hist.shape == (1, 3 + len(est_alpha), 3) and m.M.shape == (m.Nq, m.N)
    psi = m.time_integrate(Nt=5)[0]
    m.close()  # one pool at a time: a second pool in a process that owns one can deadlock (fork)
    assert np.array_equal(psi, ref.time_integrate(Nt=5)[0])
    ref.close()
    if est_alpha:  # psi0 with the state alone keeps the parameter rows
        rho = m.current_state[3:].copy()
        m.reset_model(psi0=np.ones((3, 3)))
        assert np.array_equal(m.current_state, np.vstack((np.ones((3, 3)), rho)))
        with pytest.raises(ValueError, match='are estimated'):  # the rows, not alpha0, set rho
            m.reset_model(rho=35.)
    else:  # a new parameter renames the ensemble
        m.reset_model(rho=35.)
        assert m.filename == 'Lorenz63_rho35.0_ensemble_m3'
    with pytest.raises(ValueError, match='psi0 has 1 member'):
        m.reset_model(psi0=np.ones(3))
    m.close()


def test_reset_keeps_the_integrator():
    v = VdP()
    v.integrator.method, v.integrator.max_steps = 'LSODA', 50
    v.reset_model()
    assert type(v.integrator).__name__ == 'IVPIntegrator'
    assert (v.integrator.method, v.integrator.max_steps) == ('LSODA', 50)
    v.close()
    dt = 0.1 / (16 / 71)  # a step that the dt setter of Model rounds, and KS keeps exactly
    k = advanced(KS(Nx=32, dt=dt, L=10.0), Nt=5)
    x = k.current_state.copy()
    k.reset_model()
    assert type(k.integrator).__name__ == 'DiscreteIntegrator' and k.dt == dt
    k.reset_model(dt=0.05)
    ref = KS(Nx=32, dt=0.05, L=10.0, psi0=x)
    assert np.array_equal(k.time_integrate(Nt=5)[0], ref.time_integrate(Nt=5)[0])  # ETDRK4 rebuilt for dt


def test_reset_ks2d_parameter_rebuilds_operator():
    kw = dict(Nx=32, Ny=32, dt=0.1)
    m = KS2D(nu1=0.5, nu2=0.35, **kw)
    x = m.current_state.copy()
    m.time_integrate(Nt=1)  # caches the ETDRK4 coefficients of nu2 = 0.35
    m.reset_model(nu2=0.3)
    ref = KS2D(nu1=0.5, nu2=0.3, psi0=x, **kw)
    assert m.aniso == ref.aniso
    np.testing.assert_allclose(m.time_integrate(Nt=5)[0], ref.time_integrate(Nt=5)[0], rtol=0, atol=1e-13)
    with pytest.raises(ValueError, match='must be positive'):
        m.reset_model(nu1=0.)
    assert m.nu1 == 0.5


def test_reset_rejects_bad_input_and_leaves_model_intact():
    m = advanced(Lorenz63())
    before = deepcopy(vars(m))
    with pytest.raises(ValueError, match='Time step must be positive'):
        m.reset_model(dt=-1.)
    with pytest.raises(ValueError, match="not \\['rhoo'\\]"):
        m.reset_model(rhoo=30.)
    with pytest.raises(ValueError, match='must have 3 rows'):
        m.reset_model(psi0=np.ones(4))
    assert m.dt == before['_dt'] and m.alpha0 == before['_alpha0'] and m.psi0 is not None
    assert np.array_equal(m.hist, before['history'].hist) and np.array_equal(m.hist_t, before['history'].hist_t)
    advanced(m, Nt=2)  # still integrates
    m.close()
    l96 = Lorenz96(Nx=10)
    with pytest.raises(ValueError, match="not \\['Nx'\\]"):
        l96.reset_model(Nx=20)
    l96.close()


def kept(model, before):
    """True if every attribute of `model` holds the object that it held in `before` (a copy of `vars(model)`)."""
    return vars(model).keys() == before.keys() and all(vars(model)[key] is val for key, val in before.items())


@pytest.mark.parametrize('cached', [True, False], ids=['filename-cached', 'filename-not-cached'])
@pytest.mark.parametrize('make, override, error, match', [
    (Lorenz63, dict(rho=None), TypeError, 'take real numbers'),     # t_lyap derives from rho
    (Lorenz63, dict(rho='35'), TypeError, 'take real numbers'),
    (Rijke, dict(beta=None), TypeError, 'take real numbers'),
    (VdP, dict(beta='a'), TypeError, 'take real numbers'),          # nothing but the filename derives from beta
    (lambda: Lorenz96(Nx=40), dict(F=None), TypeError, 'take real numbers'),   # t_lyap derives from F if Nx = 10
    (Lorenz63, dict(sigma=True), TypeError, 'take real numbers'),   # a bool is an int, not a parameter value
    (VdP, dict(beta=1j), TypeError, 'take real numbers'),           # a complex number is not a parameter value
    (Lorenz63, dict(dt='a'), TypeCheckError, 'float'),
    (Lorenz63, dict(dt=True), TypeError, 'must be a real number'),  # a numpy UFuncTypeError where precision_t derives
    (Lorenz63, dict(dt=np.inf), ValueError, 'positive and finite'),     # an OverflowError where precision_t derives
    (Lorenz63, dict(dt=np.nan), ValueError, 'positive and finite')],
    ids=['Lorenz63-rho-None', 'Lorenz63-rho-str', 'Rijke-beta-None', 'VdP-beta-str', 'Lorenz96-F-None',
         'Lorenz63-sigma-bool', 'VdP-beta-complex', 'dt-str', 'dt-bool', 'dt-inf', 'dt-nan'])
def test_reset_with_a_bad_value_leaves_the_model_unchanged(make, override, error, match, cached):
    """A reset with a value that is not a real number, or a dt that is not a time step, raises a clear error and leaves
    every attribute of the model as it was, and the model forecasts as before. The outcome does not depend on whether
    the filename, which a new parameter derives again, is cached: without the cache, VdP(beta='a') and
    Lorenz96(Nx=40, F=None) derive nothing from the value, and a reset accepted it."""
    m = advanced(make(), Nt=3)
    if cached:
        _ = m.filename
    ref = restarted(m)
    before, history, hist = dict(vars(m)), dict(vars(m.history)), m.hist.copy()
    with pytest.raises(error, match=match):
        m.reset_model(**override)
    assert kept(m, before) and kept(m.history, history) and np.array_equal(m.hist, hist)
    assert m.integrator.model is m
    m.reset_model()  # still resets, and forecasts as a copy made before the failure
    assert np.array_equal(m.time_integrate(Nt=3)[0], ref.time_integrate(Nt=3)[0])
    m.close(), ref.close()


def test_reset_derived_that_fails_is_undone():
    """An error in reset_derived of a subclass undoes Model.__init__, which has already set the new parameter."""
    class Failing(Lorenz63):
        def reset_derived(self, dt, **overrides):
            super().reset_derived(dt, **overrides)
            raise RuntimeError('cannot derive')

    m = Failing()
    before = dict(vars(m))
    with pytest.raises(RuntimeError, match='cannot derive'):
        m.reset_model(rho=35., dt=0.01)
    assert kept(m, before) and (m.rho, m.alpha0['rho'], m.dt, m.t_lyap) == (28., 28., 0.02, Lorenz63().t_lyap)


def test_reset_interrupted_by_a_base_exception_is_undone():
    """The undo covers any BaseException, e.g. a KeyboardInterrupt (Ctrl-C) while reset_derived runs, not only the
    subclasses of Exception."""
    class Interrupted(Lorenz63):
        def reset_derived(self, dt, **overrides):
            super().reset_derived(dt, **overrides)
            raise KeyboardInterrupt

    m = advanced(Interrupted(), Nt=3)
    _ = m.filename
    integrator, before, history, hist = m.integrator, dict(vars(m)), dict(vars(m.history)), m.hist.copy()
    with pytest.raises(KeyboardInterrupt):
        m.reset_model(rho=35., dt=0.01)
    assert kept(m, before) and kept(m.history, history) and np.array_equal(m.hist, hist)
    assert m.integrator is integrator and (m.rho, m.alpha0['rho'], m.dt) == (28., 28., 0.02)
    m.close()


def test_rijke_reset_of_tau_that_fails_later_keeps_the_limits_of_tau():
    """For a new tau, Rijke.reset_derived lengthens the delay line and sets new limits for tau. A failure after it
    restores the limits, which reset_derived replaces with a new dict rather than changing in place, and the span of
    the delay line."""
    class FailingLater(Rijke):
        def reset_derived(self, dt, **overrides):
            super().reset_derived(dt, **overrides)
            raise RuntimeError('after Rijke.reset_derived')

    m = FailingLater()
    lims = m.alpha_lims
    saved, before = dict(lims), dict(vars(m))
    with pytest.raises(RuntimeError, match='after Rijke.reset_derived'):
        m.reset_model(tau=2e-3)
    assert kept(m, before) and m.alpha_lims is lims and lims == saved and lims['tau'] == (1e-6, 1.5e-3)
    assert (m.tau, m.tau_adv, m.governing_eqns_params['tau_adv']) == (1.5e-3, 1.5e-3, 1.5e-3)
    m.close()


def test_rejected_reset_keeps_the_integrator_and_pool_of_an_ensemble():
    m = Lorenz63()
    m.init_ensemble(m=3, std_phi=0.1, est_alpha=['rho'], std_alpha=0.1)
    advanced(m, Nt=5)
    integrator, pool = m.integrator, m.integrator._pool
    assert pool is not None
    before, hist, filename = dict(vars(m)), m.hist.copy(), m.filename
    for bad, error in ((dict(sigma='a'), TypeError),  # not a real number
                       (dict(beta=None), TypeError),
                       (dict(dt='a'), TypeCheckError),
                       (dict(dt=-1.), ValueError),
                       (dict(rho=30.), ValueError),  # rho is estimated
                       (dict(psi0=np.ones((3, 2))), ValueError)):
        with pytest.raises(error):
            m.reset_model(**bad)
        assert kept(m, before) and m.integrator is integrator and integrator._pool is pool
        assert np.array_equal(m.hist, hist) and m.filename == filename and m.alpha0['sigma'] == 10.
    advanced(m, Nt=2)  # the live pool still serves the ensemble
    assert integrator._pool is pool
    m.close()


@pytest.mark.parametrize('averaged', [False, True], ids=['members', 'averaged'])
def test_parameter_override_on_an_ensemble_that_estimates_a_parameter(averaged):
    """A reset that overrides sigma and dt of an ensemble that estimates rho forecasts as a new model with the same
    sigma and dt, initialized with the same members."""
    m = Lorenz63()
    m.init_ensemble(m=3, std_phi=0.1, est_alpha=['rho'], std_alpha=0.1)
    advanced(m, Nt=5)
    x = m.current_state.copy()
    m.reset_model(sigma=11., dt=0.01)
    ref = Lorenz63(sigma=11., dt=0.01)
    ref.init_ensemble(m=3, est_alpha=['rho'], ensemble_psi0=x)
    assert (m.dt, m.alpha0, m.filename) == (ref.dt, ref.alpha0, ref.filename) == \
        (0.01, dict(rho=28., sigma=11., beta=8. / 3.), 'Lorenz63_sigma11.0_ensemble_m3')
    a, ta = m.time_integrate(Nt=10, averaged=averaged)
    m.close()  # one pool at a time
    b, tb = ref.time_integrate(Nt=10, averaged=averaged)
    ref.close()
    assert np.array_equal(a, b) and np.array_equal(ta, tb)
    np.testing.assert_allclose(a[:, 3], np.repeat(x[np.newaxis, 3], 10, axis=0), rtol=1e-14)  # each member keeps rho


def lorenz63_ensemble():
    """A Lorenz63 ensemble of 3 members that estimates rho."""
    m = Lorenz63()
    m.t_transient = 1.
    m.init_ensemble(m=3, std_phi=0.1, est_alpha=['rho'], std_alpha=0.1)
    return m


@pytest.mark.parametrize('make, override', [(Lorenz63, dict(rho=35., dt=0.01)),
                                            (lorenz63_ensemble, dict(beta=3.)),
                                            (lambda: KS(Nx=32, L=22., dt=0.1), dict(dt=0.05))],
                         ids=['Lorenz63', 'Lorenz63-ensemble', 'KS'])
def test_pickle_and_deepcopy_of_a_reset_model(make, override):
    """A reset model survives pickle and deepcopy: the copies forecast as the model, and reset as the model does."""
    import pickle

    def forecasts_alike(models):
        psi = []
        for model in models:
            psi.append(model.time_integrate(Nt=5)[0])
            model.close()  # one pool at a time
        return all(np.array_equal(p, psi[0]) for p in psi)

    m = advanced(make(), Nt=3)
    m.reset_model(**override)
    copies = [pickle.loads(pickle.dumps(m)), deepcopy(m)]
    for copy in copies:
        assert copy.integrator.model is copy and (copy.alpha0, copy.dt, copy.filename) == (m.alpha0, m.dt, m.filename)
    assert forecasts_alike([m, *copies])
    for model in (m, *copies):  # the copies reset as the original does
        model.reset_model(dt=0.02)
    assert forecasts_alike([m, *copies])


def test_reset_rijke_ensemble_that_estimates_tau():
    """Estimating tau lengthens the delay line in modify_settings (Nc = 50, tau_adv = 1e-2): a reset keeps the line,
    and forecasts as a new model with the same parameters and members."""
    m = Rijke()
    m.t_transient = 0.
    m.init_ensemble(m=3, std_phi=0.05, est_alpha=['tau', 'beta'], std_alpha=0.01)
    assert (m.Nc, m.Nphi, m.tau_adv) == (50, 70, 1e-2)
    advanced(m, Nt=3)
    x = m.current_state.copy()
    m.reset_model(C1=0.06)
    assert (m.Nc, m.Nphi, m.tau_adv, m.alpha_lims['tau']) == (50, 70, 1e-2, (1e-6, 1e-2))
    assert (m.governing_eqns_params['Nc'], m.governing_eqns_params['tau_adv']) == (50, 1e-2)
    ref = Rijke(C1=0.06)
    ref.init_ensemble(m=3, est_alpha=['tau', 'beta'], ensemble_psi0=x)
    assert m.filename == ref.filename == 'Rijke_C10.06_Nc50_ensemble_m3' and m.alpha0 == ref.alpha0
    a = m.time_integrate(Nt=3)[0]
    m.close()
    b = ref.time_integrate(Nt=3)[0]
    ref.close()
    assert np.array_equal(a, b)
