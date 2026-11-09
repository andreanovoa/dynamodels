"""Smoke tests for the physical models + the package's torch-free guarantee."""
import sys

import numpy as np
import pytest

import dynamodels
from dynamodels.physical import KS, KS2D, Annular, Kuznetsov, Lorenz63, Lorenz96, Rijke, VdP


def test_torch_free_import():
    # the whole point of the split: importing the modelling core must not drag
    # in any ML stack
    assert 'torch' not in sys.modules, 'importing dynamodels pulled in torch'
    assert 'skopt' not in sys.modules, 'importing dynamodels pulled in scikit-optimize'
    assert dynamodels.__version__


def test_models_integrate_smoke():
    for cls, kwargs in [(Lorenz63, {}), (Lorenz96, dict(Nx=10)), (VdP, {}), (Kuznetsov, {}),
                        (Rijke, {}), (Annular, {}), (KS, {}),
                        (KS2D, dict(Nx=32, Ny=32, nu1=0.5, nu2=0.35, dt=0.1))]:
        model = cls(**kwargs)
        psi, t = model.time_integrate(Nt=10)
        assert psi.shape[0] == len(t) and np.all(np.isfinite(psi)), f'{cls.__name__} integration failed'
        model.update_history(psi, t)
        y = model.get_observable_hist()
        assert y.shape[1] == model.Nq, f'{cls.__name__} observables wrong shape'
        assert len(model.obs_labels) == model.Nq
        model.close()


def test_ks2d_cases():
    # defaults are the periodic regime; case= merges a named regime, and
    # explicit kwargs win over the case values
    m = KS2D()
    assert (m.nu1, m.nu2, m.Nx, m.Ny, m.dt) == (0.5, 0.2, 32, 32, 0.01)
    c = KS2D(case='chaotic')
    assert (c.nu1, c.Nx, c.t_transient, c.t_CR) == (0.1, 64, 40., 4.)
    assert c.Lhat.shape == (64, 64), 'case must be applied before the operator is built'
    o = KS2D(case='chaotic', Nx=128, Ny=128)
    assert o.Nx == 128 and o.nu1 == 0.1
    with pytest.raises(ValueError):
        KS2D(case='nope')
    for mm in (m, c, o):
        mm.close()


def test_annular_cases():
    # explicit kwargs win over the case values
    for case, (nu, c2beta) in [('spinning', (30., 5.)), ('standing', (0., 50.)),
                               ('mixed', (20., 18.))]:
        m = Annular(case=case)
        assert (m.nu, m.c2beta) == (nu, c2beta)
        m.close()
    o = Annular(case='spinning', nu=99.)
    assert o.nu == 99. and o.c2beta == 5.
    o.close()
    with pytest.raises(ValueError):
        Annular(case='nope')


def test_rijke_cases():
    for case, beta in [('limit_cycle', 4.0), ('frequency_locked', 8.0),
                       ('chaotic', 12.0), ('relaminarized', 18.0)]:
        m = Rijke(case=case)
        assert m.beta == beta
        m.close()
    o = Rijke(case='chaotic', beta=99.)
    assert o.beta == 99.
    o.close()
    with pytest.raises(ValueError):
        Rijke(case='nope')


def test_rijke_delay_longer_than_the_default():
    """The delay line spans a delay tau longer than the default 1.5e-3, given to the constructor or to reset_model,
    and keeps the default span for a shorter delay."""
    m = Rijke(tau=2e-3)
    assert m.tau_adv == m.governing_eqns_params['tau_adv'] == 2e-3 and m.alpha_lims['tau'] == (1e-6, 2e-3)
    psi = m.time_integrate(Nt=20)[0]
    assert np.all(np.isfinite(psi))
    r = Rijke()
    r.reset_model(tau=2e-3)
    assert r.tau_adv == r.governing_eqns_params['tau_adv'] == 2e-3 and r.alpha_lims['tau'] == (1e-6, 2e-3)
    assert np.array_equal(r.time_integrate(Nt=20)[0], psi)
    short = Rijke(tau=1e-3)
    r.reset_model(tau=1e-3)
    assert short.tau_adv == r.tau_adv == r.governing_eqns_params['tau_adv'] == 1.5e-3
    assert np.array_equal(r.time_integrate(Nt=20)[0], short.time_integrate(Nt=20)[0])
    for model in (m, r, short):
        model.close()


def test_rijke_delay_line_points_grow_with_the_delay():
    """Without Nc, the delay line has 5000 points per second of tau_adv, and no fewer than 10. A coarser Nc warns, and
    reset_model refuses a delay that the line of the model cannot resolve, leaving the model unchanged."""
    for tau, nc in ((1e-3, 10), (1.5e-3, 10), (2e-3, 10), (3e-3, 15), (5e-3, 25), (1e-2, 50)):
        m = Rijke(tau=tau)
        assert (m.Nc, m.Nphi, m.psi0.shape, m.Dc.shape) == (nc, 20 + nc, (20 + nc, 1), (nc + 1, nc + 1))
        m.close()
    with pytest.warns(UserWarning, match=r'Nc = 10 points over tau_adv = 0\.003, fewer than the 15'):
        coarse = Rijke(tau=3e-3, Nc=10)
    assert coarse.Nc == 10
    r = Rijke()
    with pytest.raises(ValueError, match=r'needs Nc >= 15 points on the delay line; the model has Nc = 10'):
        r.reset_model(tau=3e-3)
    assert (r.tau, r.tau_adv, r.Nc, r.alpha_lims['tau']) == (1.5e-3, 1.5e-3, 10, (1e-6, 1.5e-3))
    wide = Rijke(Nc=15)
    wide.reset_model(tau=3e-3)
    assert (wide.tau_adv, wide.Nc) == (3e-3, 15)
    for model in (coarse, r, wide):
        model.close()


def test_rijke_delay_line_converges():
    """At tau = 3e-3, the line of the default rule (Nc = 15) gives the statistics of a fine line (Nc = 40): the RMS of
    the acoustic modes over 0.3-0.6 s agrees within 2% (with Nc = 10, it is 20% off)."""
    def rms(**kwargs):
        m = Rijke(**kwargs)
        m.t_transient = 0.
        eta = m.time_integrate(Nt=6000)[0][3000:, :m.Nm]
        m.close()
        return np.sqrt(np.mean(eta ** 2))
    default, fine = rms(tau=3e-3), rms(tau=3e-3, Nc=40)
    assert abs(default - fine) / fine < 0.02


def test_rijke_estimates_tau_with_more_than_50_points():
    """An estimated tau spans 1e-2 with at least 50 points: a model with more keeps them."""
    model = Rijke(Nc=60)
    model.t_transient = 1e-3
    model.init_ensemble(m=3, std_alpha={'tau': (1e-3, 2e-3)})
    assert (model.Nc, model.tau_adv, model.Nphi, model.psi0.shape) == (60, 1e-2, 80, (80, 1))
    assert np.all(np.isfinite(model.time_integrate(Nt=5)[0]))
    model.close()


def test_rijke_derives_its_tube_from_the_keywords():
    """Nm, Nc, Nq, xf and L size the modes, the delay line and the microphones, and place the flame."""
    m = Rijke(Nm=5, Nc=20, Nq=3, xf=0.3, L=2.)
    assert (m.Nphi, m.Dc.shape, m.gc.shape) == (2 * 5 + 20, (21, 21), (21,))
    np.testing.assert_allclose(m.jpiL, np.arange(1, 6) * np.pi / 2.)
    np.testing.assert_allclose(m.sinomjxf, np.sin(m.jpiL * 0.3))
    np.testing.assert_allclose(m.x_mic, [0.3, 0.3 + 1.7 / 3, 0.3 + 3.4 / 3])
    assert m.meanFlow['u'] == 0.15 * 10 + 0.85 * 11.1643
    psi, t = m.time_integrate(Nt=20)
    m.update_history(psi, t)
    assert np.all(np.isfinite(psi)) and m.get_observables().shape == (3, 1) and m.M.shape == (3, m.N)
    m.close()


def test_rijke_xf_keys_the_filename_and_the_respawn():
    """xf changes the dynamics, so it enters the filename, which keys the caches of create_dataset and of the truth of
    romda. A model rebuilt from alpha0 and fixed_params, as ntsa.respawn rebuilds it, keeps xf and its microphones."""
    default, moved = Rijke(), Rijke(xf=0.3)
    assert default.filename == 'Rijke_default' and moved.filename == 'Rijke_default_xf0.3'
    respawned = Rijke(psi0=moved.psi0[:, 0], dt=moved.dt,
                      **{**moved.alpha0, **{key: getattr(moved, key) for key in moved.fixed_params}})
    assert respawned.xf == 0.3 and respawned.filename == moved.filename
    np.testing.assert_array_equal(respawned.x_mic, moved.x_mic)
    for model in (moved, respawned):
        psi, t = model.time_integrate(Nt=20)
        model.update_history(psi, t)
    np.testing.assert_array_equal(respawned.get_observable_hist(), moved.get_observable_hist())
    for model in (default, moved, respawned):
        model.close()


def test_rijke_length_keys_the_filename_apart_from_dynamodels_0_6_0(tmp_path):
    """dynamodels 0.6.0 wrote a length L != 1 as '_L{L}' in the filename, but built the modes and the mean flow from
    L = 1, so the datasets that it cached under those names hold other dynamics. The label Ltube keeps create_dataset
    from loading them; the names of the default length do not change."""
    from dynamodels.utils import create_dataset, save_to_mat_file
    long = Rijke(L=1.2)
    assert long.filename == 'Rijke_default_Ltube1.2' and Rijke().filename == 'Rijke_default'
    assert Rijke(L=1.2, xf=0.3, beta=3.).filename == 'Rijke_beta3.0_Ltube1.2_xf0.3'
    stale = dict(clean_data=np.zeros((5, long.Nphi)), noisy_data=np.zeros((5, long.Nphi)), t=np.arange(5.), N_lyap=200)
    save_to_mat_file(str(tmp_path / 'Rijke_default_L1.2_Nlyap1_noise0.02_seed0'), stale)     # as 0.6.0 named it
    data, path = create_dataset(Rijke, str(tmp_path), num_lyap_times=1, L=1.2)
    assert path.endswith('Rijke_default_Ltube1.2_Nlyap1_noise0.02_seed0') and np.abs(data['clean_data']).max() > 0
    long.close()


def test_rijke_tau_adv_keyword():
    """A tau_adv keyword spans the delay line if it is longer than tau, and gives way to a longer tau, as when
    ntsa.respawn carries the tau_adv of a model into a sweep over tau. reset_model keeps it, as the constructor does."""
    long = Rijke(tau=1e-3, tau_adv=1e-2, Nc=50)
    assert long.tau_adv == long.governing_eqns_params['tau_adv'] == 1e-2
    long.reset_model(tau=2e-3)
    fresh = Rijke(tau=2e-3, tau_adv=1e-2, Nc=50)
    assert long.tau_adv == fresh.tau_adv == long.governing_eqns_params['tau_adv'] == 1e-2
    assert long.alpha_lims['tau'] == (1e-6, 1e-2)
    assert np.array_equal(long.time_integrate(Nt=20)[0], fresh.time_integrate(Nt=20)[0])
    swept = Rijke(tau=2e-3, tau_adv=Rijke.tau)
    assert swept.tau_adv == swept.governing_eqns_params['tau_adv'] == 2e-3
    assert np.all(np.isfinite(swept.time_integrate(Nt=20)[0]))
    for model in (long, fresh, swept):
        model.close()


def test_rijke_estimated_tau_longer_than_the_delay_line_raises():
    """The delay line of an estimated tau spans 1e-2: a longer tau raises at init_ensemble, not at the forecast."""
    model = Rijke(tau=2e-2)
    with pytest.raises(ValueError, match=r'tau_adv = 0\.01 \(Nc = 50\), shorter than tau = 0\.02'):
        model.init_ensemble(m=3, est_alpha=['tau'])
    assert (model.Nc, model.tau_adv, model.ensemble_cfg, model.hist.shape) == (100, 2e-2, False, (1, 120, 1))
    model.close()


def test_rijke_estimates_tau_after_an_ensemble_that_estimates_beta():
    """Estimating tau lengthens the delay line of the state rows only: the parameter row of a previous ensemble stays
    out of the new initial state, psi0, which keeps Nphi rows."""
    model = Rijke()
    model.t_transient = 1e-3
    model.init_ensemble(m=3, std_alpha={'beta': (3.5, 4.5)})
    state = model.current_state[:30].mean(axis=1)
    model.init_ensemble(m=3, std_alpha={'tau': (1e-3, 2e-3)})
    assert model.psi0.shape == (70, 1) == (model.Nphi, 1) and model.hist.shape == (1, 71, 3)
    np.testing.assert_allclose(model.psi0[:30, 0], state)
    assert not model.psi0[30:].any()
    model.init_ensemble(m=3, std_alpha={'tau': (1e-3, 2e-3)})
    assert model.psi0.shape == (70, 1) and model.hist.shape == (1, 71, 3)
    model.close()


def test_ks2d_estimable_nus():
    # nu1/nu2 are estimable params with bounds; per-member alphas get their own
    # ETDRK4 operators, matching independent single-nu runs from the same IC
    m = KS2D()
    assert m.alpha0 == dict(nu1=0.5, nu2=0.2)
    assert m.alpha_lims['nu1'] == (0.01, 1.0) and m.alpha_lims['nu2'] == (0.01, 1.0)
    u0 = m.psi0
    two = KS2D(psi0=np.hstack([u0, u0]))
    alphas = [dict(nu1=0.5, nu2=0.2), dict(nu1=0.5, nu2=0.35)]
    psi, _ = two.time_step(Nt=20, alpha=alphas)
    assert not np.allclose(psi[-1, :, 0], psi[-1, :, 1]), 'different nus must diverge'
    for k, al in enumerate(alphas):
        ref = KS2D(psi0=u0.copy(), **al)
        psi_ref, _ = ref.time_step(Nt=20)
        assert np.allclose(psi[:, :, k], psi_ref[:, :, 0]), f'member {k} deviates from its single-nu run'
        ref.close()
    # augmented ensemble: estimated-parameter rows ride through time_integrate
    e = KS2D(t_transient=1.)  # keep init_ensemble's transient run short
    e.init_ensemble(m=3, std_phi=0.1, est_alpha=['nu2'], std_alpha=0.1)
    psi, t = e.time_integrate(Nt=5)
    assert psi.shape[1] == e.Nphi + 1 and np.all(np.isfinite(psi))
    assert np.allclose(psi[-1, -1], psi[0, -1]), 'estimated nu2 row must be carried unchanged'
    for mm in (m, two, e):
        mm.close()


def test_t_lyap_tables():
    # measured-lambda1 tables in the physical model files: instances get
    # t_lyap = 1/lambda1(sweep param) inside the chaotic range, class constants
    # and off-table configurations are untouched
    assert np.isclose(Lorenz63.t_lyap, 0.9056 ** -1), 'class access must stay the constant'
    ms = [Lorenz63(), Lorenz63(rho=10.),
          Lorenz96(Nx=10, F=8.), Lorenz96(Nx=10, F=4.8), Lorenz96(Nx=10, F=2.),
          Lorenz96(Nx=40, F=8.), Rijke(beta=12.), Rijke(beta=4.)]
    expect = [1 / 0.917, 0.9056 ** -1,
              1 / 1.184, None, 1.67 ** -1,
              1.67 ** -1, 1 / 161.0, 0.02]
    for m, e in zip(ms, expect):
        if e is None:  # F=4.8: log-interpolated between the F=4.6 and F=5 entries
            assert 1 / 0.058 < m.t_lyap < 1 / 0.0391, f'{m.name} t_lyap = {m.t_lyap}'
        else:
            assert np.isclose(m.t_lyap, e), f'{type(m).__name__} t_lyap = {m.t_lyap}, expected {e}'
        m.close()


def test_pickle_drops_live_pool():
    # pickling a mid-run model must not require close(): __getstate__ drops the
    # (unpicklable) live pool and the copy lazily re-creates one when needed
    import pickle
    import threading

    m = Lorenz63()
    psi, t = m.time_integrate(Nt=5)
    m.update_history(psi, t)
    m.integrator._pool = threading.Lock()  # stand-in for an unpicklable live pool
    m2 = pickle.loads(pickle.dumps(m))
    assert m2.integrator._pool is None
    assert np.allclose(m2.current_state, m.current_state)
    m.integrator._pool = None
    m.close()
    m2.close()


@pytest.mark.parametrize('m, averaged', [(1, False), (4, False), (4, True)])
def test_ks_pickle_roundtrip(m, averaged):
    # KS caches its nonlinear operator, a bound method, in ETDRK4_f_terms. pickle rebuilds a
    # bound method as getattr(obj, func.__name__), which a name-mangled (double underscore)
    # method does not resolve, so unpickling raised AttributeError (copy() rebinds by object).
    import pickle

    model = KS(Nx=64, seed=1, initial_amplitude=1.)
    if m > 1:
        model.init_ensemble(m=m, std_phi=0.01)
    psi, t = model.time_integrate(Nt=20)
    model.update_history(psi, t)
    clone = pickle.loads(pickle.dumps(model))
    assert clone.m == m
    np.testing.assert_array_equal(clone.hist, model.hist)
    np.testing.assert_array_equal(clone.hist_t, model.hist_t)
    # the cached operator must act on the clone, not on the pickled original
    assert clone.ETDRK4_f_terms['nonlinear_operator'].__self__ is clone
    pa, ta = model.time_integrate(Nt=50, averaged=averaged)
    pb, tb = clone.time_integrate(Nt=50, averaged=averaged)
    np.testing.assert_array_equal(pb, pa)
    np.testing.assert_array_equal(tb, ta)


def test_ks_pickle_of_dynamodels_0_6_0_loads():
    """dynamodels 0.6.0 cached the nonlinear operator of KS as a bound method named '__nonlinear_operator', which pickle
    rebuilds with getattr(model, '__nonlinear_operator'): an alias of the method resolves it."""
    import pickle
    import types

    model = KS(Nx=32, seed=1, initial_amplitude=1.)
    model.time_integrate(Nt=2)          # caches the operator
    terms = model.ETDRK4_f_terms
    old = types.FunctionType(KS._nonlinear_operator.__code__, KS._nonlinear_operator.__globals__,
                             '__nonlinear_operator')
    terms['nonlinear_operator'] = types.MethodType(old, model)      # as 0.6.0 cached it
    clone = pickle.loads(pickle.dumps(model))
    assert clone.ETDRK4_f_terms['nonlinear_operator'].__self__ is clone
    np.testing.assert_array_equal(clone.time_integrate(Nt=5)[0], model.time_integrate(Nt=5)[0])


def test_create_dataset(tmp_path):
    from dynamodels.utils import create_dataset
    data, path = create_dataset(Lorenz63, str(tmp_path), num_lyap_times=2, seed=1)
    assert data['clean_data'].shape == data['noisy_data'].shape
    assert data['clean_data'].shape[1] == 3
    assert len(data['t']) == len(data['clean_data'])
    assert not np.allclose(data['clean_data'], data['noisy_data'])
    # second call must hit the .mat cache, not re-integrate
    cached, path2 = create_dataset(Lorenz63, str(tmp_path), num_lyap_times=2, seed=1)
    assert path2 == path
    assert np.allclose(cached['clean_data'], data['clean_data'])
    # Nx must key the cache: a different structural size gets its own file
    d10, p10 = create_dataset(Lorenz96, str(tmp_path), num_lyap_times=2, Nx=10)
    assert p10 != path and d10['clean_data'].shape[1] == 10
    # xf changes the dynamics of Rijke without changing its size: it must key the cache too
    d02, p02 = create_dataset(Rijke, str(tmp_path), num_lyap_times=1)
    d03, p03 = create_dataset(Rijke, str(tmp_path), num_lyap_times=1, xf=0.3)
    assert p03 != p02 and not np.allclose(d03['clean_data'], d02['clean_data'])


def test_ks_etdrk4_matches_reference_integration():
    # regression for the pre-0.3.2 stepper, which used f-coefficients in the RK
    # stages and divided the final combination by 6 -- first-order consistent
    # with u_t = L u + N(u)/6 (a 6x-amplitude rescale of KS), 7% off a reference
    # integration by t = 0.5. The standard Kassam-Trefethen scheme must track an
    # integrating-factor RK4 reference at fine step to high accuracy.
    # keep |dt * L_hat| < 15 for every mode so the contour quadrature is exact
    # for all of them (far-outside modes deliberately match the dense-operator
    # stack's behavior instead -- see ETDRK4_f_terms)
    dt, L, Nx, Nt = 0.005, 22.0, 48, 100
    m = KS(Nx=Nx, dt=dt, L=L, seed=0, initial_amplitude=1.0)
    k = m.k
    Lhat = (k**2 - k**4)[:, None]

    def Nrhs(u_hat):
        u = np.fft.irfft(u_hat, axis=0)
        return -0.5 * (1j * k[:, None]) * np.fft.rfft(u**2, axis=0)

    psi, _ = m.time_integrate(Nt=Nt)
    u_dm = psi[-1]

    sub = 100
    h = dt / sub
    E1, Eh = np.exp(h * Lhat), np.exp(h / 2 * Lhat)
    u = m.psi0.copy()
    for _ in range(Nt * sub):
        k1 = Nrhs(u)
        k2 = Nrhs(Eh * (u + h / 2 * k1))
        k3 = Nrhs(Eh * u + h / 2 * k2)
        k4 = Nrhs(E1 * u + h * Eh * k3)
        u = E1 * u + h / 6 * (E1 * k1 + 2 * Eh * (k2 + k3) + k4)

    rel = np.linalg.norm(u_dm - u) / np.linalg.norm(u)
    # ~5e-5 is the scheme's genuine truncation error at this dt with an O(1)
    # rough IC; the pre-0.3.2 scheme sat at ~7e-2 regardless of dt.
    assert rel < 1e-3, f'KS ETDRK4 deviates from reference integration: rel err {rel:.2e}'


def test_ks_exact_dt_preserved():
    # Model.dt rounds to precision_t decimals; KS must keep the exact requested
    # step (dt = 0.1 * 71 / 16 arises from the visc rescaling of the qlROM cases)
    dt = 0.1 / (16 / 71)
    m = KS(Nx=32, dt=dt, L=10.0)
    assert m.dt == dt


def test_ks_dt_honored_and_param_roundtrip():
    # dt used to be silently replaced by the 0.25 default at the Model level
    m = KS(Nx=64, dt=0.1, nu=0.08)
    assert m.dt == 0.1
    _, t = m.time_integrate(Nt=5)
    assert np.isclose(t[1] - t[0], 0.1)
    # nu alone nondimensionalizes: the domain absorbs nu and the stored operator
    # is the nu = 1 one, so (Nx, nu, L) keeps describing what was integrated.
    assert m.nu == 1.0 and m.L == pytest.approx(2 * np.pi / np.sqrt(0.08))
    # (Nx, nu, L) are fixed_params so a respawn-style rebuild is bit-faithful
    m2 = KS(Nx=64, dt=0.1, nu=m.nu, L=m.L, psi0=m.psi0.copy())
    pa, _ = m.time_step(Nt=50)
    pb, _ = m2.time_step(Nt=50)
    assert np.allclose(pa, pb)


def test_ks_two_parameter_form_honors_both_nu_and_L():
    # Regression: nu used to be silently forced to 1 whenever L was given, so an
    # independent (visc, Lx) pair was inexpressible. Both must now survive, and the
    # linear operator must be the general alpha^2 - nu alpha^4.
    Nx, Lx, visc, dt = 64, 20 * np.pi, 0.37, 0.01
    m = KS(Nx=Nx, dt=dt, L=Lx, nu=visc)
    assert m.nu == visc and m.L == Lx

    alpha = 2 * np.pi * np.arange(Nx // 2 + 1) / Lx
    np.testing.assert_allclose(m.k, alpha, rtol=0, atol=1e-12)
    # E = exp(dt * Lhat) is the exact exponential of the linear operator, so it
    # exposes the operator the ETDRK4 coefficients were rebuilt from.
    expected = np.exp(dt * (alpha**2 - visc * alpha**4))[:, None]
    np.testing.assert_allclose(m.ETDRK4_f_terms['E'], expected, rtol=1e-14, atol=0)

    # The single-parameter defaults keep their nu = 1 operator
    m_L = KS(Nx=Nx, dt=dt, L=Lx)
    assert m_L.nu == 1.0
    np.testing.assert_allclose(m_L.ETDRK4_f_terms['E'],
                               np.exp(dt * (alpha**2 - alpha**4))[:, None], rtol=1e-14, atol=0)


def test_ks_two_parameter_form_matches_rescaled_construction():
    # (visc, Lx) and the old nu = 1 rescaling describe the SAME physical system:
    #   v(x', t') = sqrt(visc) u(x, t),  x = sqrt(visc) x',  t = visc t'
    # so KS(Nx, L=Lx, nu=visc, dt=dt) and KS(Nx, L=Lx/sqrt(visc), dt=dt/visc) must
    # agree after u = v / sqrt(visc) (and t = visc t').
    # The qlROM ks1d quasi-periodic parameters. dt is finer than the case's own 0.1
    # on purpose: the two constructions differ only by floating-point roundoff, and
    # this system amplifies a 1e-15 perturbation to O(1) within t ~ 30 (verified with
    # a perturbed IC on a SINGLE construction), so the comparison horizon has to stay
    # inside the roundoff-dominated regime for the equivalence itself to be visible.
    Nx, Lx, visc, dt, Nt = 128, 2 * np.pi, 16 / 71, 0.02, 300
    s = np.sqrt(visc)
    u0 = np.cos(Lx * np.arange(Nx) / Nx)

    direct = KS(Nx=Nx, dt=dt, L=Lx, nu=visc, psi0=np.fft.rfft(u0)[:, None])
    rescaled = KS(Nx=Nx, dt=dt / visc, L=Lx / s, psi0=np.fft.rfft(s * u0)[:, None])

    assert direct.dt == dt and rescaled.dt == dt / visc

    pa, ta = direct.time_step(Nt=Nt)
    pb, tb = rescaled.time_step(Nt=Nt)

    ua = KS.fourier_to_physical(pa)[:, :, 0]
    ub = KS.fourier_to_physical(pb)[:, :, 0] / s
    err = np.max(np.abs(ua - ub)) / np.max(np.abs(ua))
    assert err < 1e-10, f'rescaling equivalence broken: rel err {err:.2e}'
    # time also maps: t = visc * t' (the stamps themselves are rounded to
    # precision_t by Model, so compare the exact steps they are built from)
    assert visc * rescaled.dt == pytest.approx(direct.dt, rel=1e-14)
    assert len(ta) == len(tb) == Nt + 1


def test_lorenz96_nx10_lyapunov_table_between_onset_and_f8():
    """The Nx = 10 table must follow the measured exponents inside the chaotic range: the earlier
    three-point table log-interpolated 0.26 at F = 6.5 against a measured 0.75."""
    from dynamodels.physical import Lorenz96
    assert abs(Lorenz96(Nx=10, F=6.5).t_lyap - 1 / 0.7488) < 0.03 / 0.7488
    assert abs(Lorenz96(Nx=10, F=5.5).t_lyap - 1 / 0.4832) < 0.03 / 0.4832
    assert Lorenz96(Nx=10, F=8.0).t_lyap == 1 / 1.184          # keys the cached datasets
    assert Lorenz96(Nx=10, F=4.0).t_lyap == Lorenz96.t_lyap    # below the chaotic range: fallback
