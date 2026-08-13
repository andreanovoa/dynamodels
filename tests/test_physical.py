"""Smoke tests for the physical models + the package's torch-free guarantee."""
import sys

import numpy as np

import dynamodels
from dynamodels.physical import KS, KS2D, Annular, Lorenz63, Lorenz96, Rijke, VdP


def test_torch_free_import():
    # the whole point of the split: importing the modelling core must not drag
    # in any ML stack
    assert 'torch' not in sys.modules, 'importing dynamodels pulled in torch'
    assert 'skopt' not in sys.modules, 'importing dynamodels pulled in scikit-optimize'
    assert dynamodels.__version__


def test_models_integrate_smoke():
    for cls, kwargs in [(Lorenz63, {}), (Lorenz96, dict(Nx=10)), (VdP, {}),
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
    # (Nx, nu, L) are fixed_params so a respawn-style rebuild is bit-faithful,
    # including the L-given branch that normalizes nu to 1
    m2 = KS(Nx=64, dt=0.1, nu=m.nu, L=m.L, psi0=m.psi0.copy())
    pa, _ = m.time_step(Nt=50)
    pb, _ = m2.time_step(Nt=50)
    assert np.allclose(pa, pb)
