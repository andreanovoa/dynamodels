"""AnnularSDE and the quaternion slow flow (port of the MATLAB codes of Faure-Beaulieu & Noiray)."""
import numpy as np
import pytest

from dynamodels.physical import Annular, AnnularSDE
from dynamodels.physical.annular_sde import (
    I_Q,
    J_Q,
    K_Q,
    analytic_hypercomplex,
    bandpass_rect,
    qconj,
    qexp,
    qmod,
    qprod,
    quaternion,
    slow_flow,
    slow_flow_of_modes,
)

ER = 0.5625


def matlab_accelerations(psi, nu, c2_b, k, w, Th2):
    """-fa, -fb of Fast_SDE_solv_lin_param_v2.m, transcribed."""
    ea, dtea, eb, dteb = psi  # dynamodels order (MATLAB stores ea, eb, dtea, dteb)
    fa = (-2 * nu * dtea + k / 4 * (9 * ea ** 2 + 3 * eb ** 2) * dtea
          - c2_b / 2 * np.cos(2 * Th2) * dtea - c2_b / 2 * np.sin(2 * Th2) * dteb + (w ** 2 + 3 / 2 * k * eb * dteb) * ea)
    fb = (-2 * nu * dteb + k / 4 * (9 * eb ** 2 + 3 * ea ** 2) * dteb
          + c2_b / 2 * np.cos(2 * Th2) * dteb - c2_b / 2 * np.sin(2 * Th2) * dtea + (w ** 2 + 3 / 2 * k * ea * dtea) * eb)
    return -fa, -fb


def test_drift_matches_matlab_without_reactive_asymmetry():
    rng = np.random.default_rng(0)
    for _ in range(200):
        psi = rng.normal(0, [500, 5e6, 500, 5e6])
        nu, c2b = rng.uniform(-30, 30), rng.uniform(0, 50)
        k, w, th = rng.uniform(0.5e-4, 2e-4), 2 * np.pi * rng.uniform(1080, 1100), rng.uniform(0, np.pi)
        d = Annular.time_derivative(0., psi, nu=nu, kappa=k, c2beta=c2b, theta_b=th, omega=w, epsilon=0., theta_e=0.)
        ma, mb = matlab_accelerations(psi, nu, c2b, k, w, th)
        assert np.allclose([d[1], d[3]], [ma, mb], rtol=1e-12, atol=1e-6)


def test_quaternion_units_and_modulus():
    assert np.allclose(qprod(I_Q, J_Q), K_Q)
    assert np.allclose(qprod(J_Q, I_Q), -K_Q)
    assert np.allclose(qprod(I_Q, I_Q), quaternion(-1.))
    rng = np.random.default_rng(1)
    p, q = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))
    assert np.allclose(qmod(qprod(p, q)), qmod(p) * qmod(q))
    assert np.allclose(qprod(p, qconj(p))[:, 1:], 0)
    x = rng.uniform(-3, 3, 50)
    assert np.allclose(qexp(qprod(I_Q, x)), quaternion(np.cos(x), np.sin(x)))


def test_slow_flow_round_trip():
    rng = np.random.default_rng(2)
    N = 400
    A = rng.uniform(1, 1000, N)
    alpha = rng.uniform(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, N)
    alpha.sort()  # no jumps above pi/2 between neighbours
    chi = rng.uniform(-np.pi / 4 + 0.01, np.pi / 4 - 0.01, N)
    phi = rng.uniform(-np.pi + 0.01, np.pi - 0.01, N)
    sig = qprod(A, qexp(qprod(I_Q, alpha)), qexp(qprod(-chi, K_Q)), qexp(qprod(J_Q, phi)))
    A2, alpha2, chi2, phi2 = slow_flow(sig)
    assert np.allclose(A2, A)
    assert np.allclose(chi2, chi, atol=1e-10)
    # the decomposition is unique up to (alpha, phi) -> (alpha + pi, phi + pi); check the reconstruction
    rec = qprod(A2, qexp(qprod(I_Q, alpha2)), qexp(qprod(-chi2, K_Q)), qexp(qprod(J_Q, phi2)))
    assert np.allclose(rec, sig, atol=1e-8)


def test_standing_and_spinning_nature_angle():
    t = np.arange(0, 0.2, 1 / 51200)
    w = 2 * np.pi * 1090
    standing = slow_flow(analytic_hypercomplex(np.cos(w * t), 0 * t))
    spinning = slow_flow(analytic_hypercomplex(np.cos(w * t), np.sin(w * t)))
    k = slice(1000, -1000)
    assert np.allclose(standing[2][k], 0, atol=1e-3)
    assert np.allclose(np.abs(spinning[2][k]), np.pi / 4, atol=1e-3)


def test_bandpass_rect_keeps_band_only():
    fs, n = 51200., 51200
    t = np.arange(n) / fs
    x = np.cos(2 * np.pi * 1090 * t) + np.cos(2 * np.pi * 1500 * t) + 3.
    y = bandpass_rect(x, 1065, 1115, fs)
    assert np.allclose(y, np.cos(2 * np.pi * 1090 * t), atol=1e-6)


def test_gamma_from_ER_matches_matlab_chain():
    w = 2 * np.pi * 1090
    Gam_on_w2 = 3.75e+06 * ER - 1.6436e+06
    sigma = np.sqrt(Gam_on_w2 / 4 / w)
    assert np.isclose(AnnularSDE.Gamma_from_ER(ER, w), 4 * sigma ** 2 * w ** 3)


def test_ou_stationary_variance():
    """Without restoring force on xi, the substep recursion is an AR(1) whose variance is known exactly."""
    model = AnnularSDE(ER=ER, nu=Annular.nu_from_ER(ER), c2beta=Annular.c2beta_from_ER(ER), seed=3)
    model.init_ensemble(m=4000, std_phi=0.)
    model.time_step(Nt=40)  # 400 substeps = 16 tc
    h, tc = model.dt_sde, model.tc
    G = AnnularSDE.ou_drive_intensity(model.Gamma, model.omega, tc, model.bandwidth)
    a, b2 = 1 - h / tc, G / tc ** 2 * h
    var_exact = b2 / (1 - a ** 2) * (1 - a ** 800)
    assert np.isclose(np.var(model.xi), var_exact, rtol=0.05)


def test_seeded_and_integrates():
    kw = dict(ER=ER, nu=Annular.nu_from_ER(ER), c2beta=Annular.c2beta_from_ER(ER), seed=7)
    runs = []
    for _ in range(2):
        model = AnnularSDE(**kw)
        psi, t = model.time_integrate(200)
        runs.append(psi)
    assert psi.shape == (200, 4, 1) and np.isclose(t[1] - t[0], model.dt)
    assert np.array_equal(runs[0], runs[1])
    assert np.all(np.isfinite(psi))


def test_noise_grows_a_stable_mode_to_the_noise_level():
    """A linearly stable mode (nu < 0) stays near the origin without noise and is excited by it."""
    kw = dict(nu=-20., c2beta=10., psi0=np.zeros(4), seed=0)
    quiet = AnnularSDE(Gamma=1e-30, **kw).time_integrate(2000)[0]
    noisy = AnnularSDE(Gamma=1e13, **kw).time_integrate(2000)[0]
    assert np.abs(quiet[:, 0]).max() < 1e-6 < np.abs(noisy[:, 0]).max()


def test_slow_flow_of_modes_removes_fast_phase():
    dt = 1 / 51200
    t = np.arange(0, 0.5, dt)
    f0 = 1090.
    A, alpha, chi, phi = slow_flow_of_modes(np.cos(2 * np.pi * f0 * t), 0 * t, dt, f0=f0, t=t)
    k = slice(2000, -2000)
    assert np.allclose(A[k], 1, atol=1e-2)
    assert np.ptp(np.unwrap(2 * phi[k])) < 0.05


@pytest.mark.parametrize('bad', [dict(tc=1e-6), dict(tc=1e-3)])
def test_timescale_warning(bad):
    with pytest.warns(UserWarning, match='correlation time'):
        AnnularSDE(**bad)
