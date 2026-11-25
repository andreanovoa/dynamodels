r"""Stochastic annular combustor and the quaternion slow flow of its azimuthal mode.

Port of the MATLAB codes of Faure-Beaulieu & Noiray (CAPS Lab, ETH Zurich) used in
Indlekofer, Faure-Beaulieu, Dawson & Noiray (2022), J. Fluid Mech. 944, A15:
`Fast_SDE_solv_lin_param_v2.m` (the stochastic fast flow) becomes `AnnularSDE`;
`BP_filter_rect.m`, the quaternion toolbox (`Quaternion.m`, `Qprod.m`, `Qexp.m`, `Qmod.m`, ...)
and `slowflow.m` become `bandpass_rect`, `qprod`, `qexp`, `qmod` and `slow_flow`.
"""
import numpy as np

from ..integrator import DiscreteIntegrator
from .annular import Annular


class AnnularSDE(Annular):
    r"""Annular combustor driven by turbulent noise: the stochastic fast flow of `Annular`.

    The deterministic right-hand side of `Annular` is forced on each mode by an independent
    noise $\xi_a$, $\xi_b$,

    $$
    \ddot{\eta}_a = f_a(\eta_a, \eta_b, \dot{\eta}_a, \dot{\eta}_b) + \xi_a, \qquad
    \ddot{\eta}_b = f_b(\eta_a, \eta_b, \dot{\eta}_a, \dot{\eta}_b) + \xi_b ,
    $$

    where $f_a$, $f_b$ are the accelerations of `Annular.time_derivative`. The white noise of
    intensity $\Gamma$, $\langle \xi_i(t) \xi_j(t+\tau) \rangle = \Gamma \delta_{ij} \delta(\tau)$,
    is replaced by an Ornstein-Uhlenbeck process of correlation time $\tau_c$
    ($\Delta t_\mathrm{SDE} \ll \tau_c \ll 2\pi/\omega$),

    $$
    \mathrm{d}\xi = -\frac{\xi}{\tau_c} \mathrm{d}t + \frac{\sqrt{\Gamma_\mathrm{OU}}}{\tau_c} \mathrm{d}W,
    \qquad
    \Gamma_\mathrm{OU} = \frac{\Gamma}{g} \left(1 + \omega^2 \tau_c^2\right),
    \qquad
    g = \frac{2\pi \tau_c B}{\arctan\!\big(2\pi (f_0 + B/2) \tau_c\big) - \arctan\!\big(2\pi (f_0 - B/2) \tau_c\big)},
    $$

    so that the coloured noise has the intensity $\Gamma$ in the band of width $B$ around
    $f_0 = \omega / 2\pi$, and the system is integrated with Heun's scheme (second order), which
    needs no Rossler method because the forcing is smooth. Each output step $\Delta t$ is split
    into the smallest number of equal substeps no longer than `dt_sde_max`.

    The noise intensity follows the linear fit of the system identification of Indlekofer et
    al. (2022), $\Gamma / \omega^2 = \Gamma_1 \Phi + \Gamma_2$ (`Gamma_from_ER`), unless `Gamma`
    is given. With $\epsilon = 0$ the model is the one of the MATLAB code, which has no reactive
    asymmetry.

    The noise is drawn from `rng` (seeded by `seed`); the OU state of every member is kept in
    `xi` (shape ``(2, m)``) between calls, and the model is advanced by `time_step`
    (`DiscreteIntegrator`), never by an IVP solver.

    References
    ----------
    Indlekofer, Faure-Beaulieu, Dawson & Noiray (2022). Spontaneous and explicit symmetry
    breaking of thermoacoustic eigenmodes in imperfect annular geometries. *J. Fluid Mech.*,
    944, A15. [DOI: 10.1017/jfm.2022.468](https://doi.org/10.1017/jfm.2022.468).
    """

    Gam_on_w2_1, Gam_on_w2_2 = 3.75e+06, -1.6436e+06  # Gamma / omega^2 = Gam_on_w2_1 * ER + Gam_on_w2_2

    Gamma = None          # noise intensity [Pa^2 s^-3]; None: Gamma_from_ER(ER, omega)
    tc = 5e-5             # correlation time of the OU noise [s]
    bandwidth = 50.       # band around f0 in which the OU noise has the intensity Gamma [Hz]
    dt_sde_max = 2e-6     # longest substep of the Heun scheme [s]

    def __init__(self, **model_dict):
        if model_dict.get('Gamma') is None:
            ER = model_dict.get('ER', self.ER)
            omega = model_dict.get('omega', self.omega)
            model_dict['Gamma'] = self.Gamma_from_ER(ER, omega)
        super().__init__(**model_dict)
        self.integrator = DiscreteIntegrator(self)
        self.xi = np.zeros((2, self.psi0.shape[-1]))
        self._check_timescales()

    @property
    def dt_step(self):
        """float: Time step seen by `DiscreteIntegrator`; `time_step` substeps it internally."""
        return self.dt

    @property
    def n_sub(self):
        """int: Number of Heun substeps per output step `dt`."""
        return int(np.ceil(self.dt / self.dt_sde_max - 1e-9))

    @property
    def dt_sde(self):
        """float: Substep of the Heun scheme, ``dt / n_sub``."""
        return self.dt / self.n_sub

    @staticmethod
    def Gamma_from_ER(ER, omega=Annular.omega):
        r"""Noise intensity $\Gamma = \omega^2 (\Gamma_1 \Phi + \Gamma_2)$ of Indlekofer et al. (2022)."""
        return (AnnularSDE.Gam_on_w2_1 * ER + AnnularSDE.Gam_on_w2_2) * omega ** 2

    @staticmethod
    def ou_drive_intensity(Gamma, omega, tc, bandwidth):
        r"""Intensity $\Gamma_\mathrm{OU}$ of the white noise that drives the OU process (see the class docstring)."""
        f0 = omega / (2 * np.pi)
        g = 2 * np.pi * tc * bandwidth / (np.arctan(2 * np.pi * (f0 + bandwidth / 2) * tc)
                                          - np.arctan(2 * np.pi * (f0 - bandwidth / 2) * tc))
        return Gamma / g * (1 + omega ** 2 * tc ** 2)

    def _check_timescales(self):
        omega = self.alpha0['omega']
        if self.dt_sde > self.tc / 10 or 10 * self.tc > 2 * np.pi / omega:
            import warnings
            warnings.warn('AnnularSDE: the noise correlation time tc must be large compared to the substep '
                          f'dt_sde = {self.dt_sde:.3g} s and small compared to the acoustic period '
                          f'{2 * np.pi / omega:.3g} s (tc = {self.tc:.3g} s)', stacklevel=3)

    def _drift(self, psi, alpha):
        """Accelerations of `Annular.time_derivative` for every member, psi (4, m)."""
        return np.array(Annular.time_derivative(0., psi, **alpha)[:4])

    def time_step(self, Nt=10, averaged=False, alpha=None):
        """Advance the stochastic system by `Nt` output steps.

        Parameters
        ----------
        Nt : int
            Number of output steps of length `dt` (each made of `n_sub` Heun substeps).
        averaged, alpha
            Accepted for interface compatibility; unused (every member is forced by its own noise).

        Returns
        -------
        psi : ndarray, shape ``(Nt + 1, N, m)``
            State at the output times, including the initial one (parameters are carried unchanged).
        t : ndarray, shape ``(Nt + 1,)``
            Output times.
        """
        psi0 = self.current_state
        if psi0.ndim == 1:
            psi0 = psi0[:, None]
        m = psi0.shape[-1]
        if self.xi.shape[-1] != m:
            self.xi = np.zeros((2, m))

        alphas = self.get_alpha()
        # one parameter set per member, as arrays of shape (m,), so the drift is evaluated for all at once
        alpha = {key: np.array([a[key] for a in alphas]) for key in alphas[0]}
        alpha = {key: alpha[key] for key in ['nu', 'kappa', 'c2beta', 'theta_b', 'omega', 'epsilon', 'theta_e']}

        Gamma_OU = self.ou_drive_intensity(self.Gamma, alpha['omega'], self.tc, self.bandwidth)
        h, tc = self.dt_sde, self.tc
        a_ou = h / tc
        b_ou = np.sqrt(Gamma_OU) / tc * np.sqrt(h)

        y = psi0[:4].copy()
        xi = self.xi
        out = np.empty((Nt + 1, 4, m))
        out[0] = y
        rng = self.rng
        for n in range(Nt):
            for _ in range(self.n_sub):
                xi_new = xi - xi * a_ou + b_ou * rng.standard_normal((2, m))
                g1 = self._drift(y, alpha)
                g1[1] += xi[0]
                g1[3] += xi[1]
                y2 = y + h * g1
                g2 = self._drift(y2, alpha)
                g2[1] += xi_new[0]
                g2[3] += xi_new[1]
                y = y + h / 2 * (g1 + g2)
                xi = xi_new
            out[n + 1] = y
        self.xi = xi

        psi = np.concatenate([out, np.broadcast_to(psi0[4:], (Nt + 1,) + psi0[4:].shape)], axis=1)
        t = np.round(self.current_time + np.arange(Nt + 1) * self.dt, self.precision_t)
        return psi, t


# %% ============================ QUATERNIONS (arrays of shape (N, 4)) ============================ %% #

I_Q = np.array([[0., 1., 0., 0.]])
J_Q = np.array([[0., 0., 1., 0.]])
K_Q = np.array([[0., 0., 0., 1.]])


def quaternion(a, b=0., c=0., d=0.):
    """Quaternion array a + i b + j c + k d of shape (N, 4); scalars broadcast to the common length."""
    a, b, c, d = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, dtype=float)).ravel() for x in (a, b, c, d)])
    return np.stack([a, b, c, d], axis=1)


def _as_quaternion(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[1] == 4:
        return x
    return quaternion(x)  # a real array is a purely real quaternion


def qprod(*args):
    """Element-wise Hamilton product of quaternion arrays (real arrays are real quaternions), left to right."""
    out = _as_quaternion(args[0])
    for x in args[1:]:
        B = _as_quaternion(x)
        A = out
        out = np.stack([A[:, 0] * B[:, 0] - A[:, 1] * B[:, 1] - A[:, 2] * B[:, 2] - A[:, 3] * B[:, 3],
                        A[:, 0] * B[:, 1] + A[:, 1] * B[:, 0] + A[:, 2] * B[:, 3] - A[:, 3] * B[:, 2],
                        A[:, 0] * B[:, 2] + A[:, 2] * B[:, 0] - A[:, 1] * B[:, 3] + A[:, 3] * B[:, 1],
                        A[:, 0] * B[:, 3] + A[:, 3] * B[:, 0] + A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1]], axis=1)
    return out


def qconj(A):
    """Conjugate quaternions."""
    return _as_quaternion(A) * np.array([1., -1., -1., -1.])


def qmod(A):
    """Modulus of each quaternion."""
    return np.sqrt(np.sum(_as_quaternion(A) ** 2, axis=1))


def qexp(A):
    """exp(a0) exp(i a1) exp(j a2) exp(k a3), as `Qexp.m` (the product of the exponentials of each part)."""
    A = _as_quaternion(A)
    z = np.zeros(len(A))
    return qprod(np.exp(A[:, 0]),
                 quaternion(np.cos(A[:, 1]), np.sin(A[:, 1]), z, z),
                 quaternion(np.cos(A[:, 2]), z, np.sin(A[:, 2]), z),
                 quaternion(np.cos(A[:, 3]), z, z, np.sin(A[:, 3])))


def bandpass_rect(sig, f_low, f_high, fs):
    """Rectangular band-pass in the Fourier domain, as `BP_filter_rect.m` (the same bin edges).

    `sig` has time along axis 0; `f_low` may be None for a low-pass filter.
    """
    sig = np.asarray(sig, dtype=float)
    n = sig.shape[0]
    spec = np.fft.fft(sig, axis=0)
    f0v = fs / 2 * np.linspace(0, 1, n // 2)

    def edge(f):  # 1-based index k with f0v(k) < f <= f0v(k + 1)
        return int(np.flatnonzero((f0v[1:] >= f) & (f0v[:-1] < f))[0]) + 1

    i_lp = edge(f_high)
    spec[i_lp:n - i_lp + 1] = 0
    if f_low is not None:
        i_hp = edge(f_low)
        spec[:i_hp - 1] = 0
        spec[n - i_hp + 2:] = 0
    return np.real(np.fft.ifft(spec, axis=0))


def analytic_hypercomplex(eta_a, eta_b):
    r"""Analytic hypercomplex signal $\eta_a + i \eta_b + j H[\eta_a] + k H[\eta_b]$ (Ghirardo & Bothien 2018).

    $H$ is the Hilbert transform; the signal is
    ``(ea + j H[ea]) + i (eb + j H[eb])`` of `Fast_SDE_solv_lin_param_v2.m`.
    """
    from scipy.signal import hilbert
    ha, hb = np.imag(hilbert(eta_a)), np.imag(hilbert(eta_b))
    return quaternion(eta_a, eta_b, ha, hb)


def slow_flow(signal):
    r"""Slow-flow variables of an azimuthal mode from its analytic hypercomplex signal (`slowflow.m`).

    The signal is $A\, e^{i \vartheta} e^{-k \chi} e^{j \varphi}$ (Ghirardo & Bothien 2018):
    $A$ the amplitude, $\vartheta$ the orientation of the anti-nodal line (`alpha`, the
    MATLAB name, $n\theta_0$), $\chi$ the nature angle ($\pm\pi/4$ spinning, 0 standing) and
    $\varphi$ the temporal phase (it still contains $\omega t$).

    Parameters
    ----------
    signal : ndarray, shape (N, 4)
        Analytic hypercomplex signal, e.g. from `analytic_hypercomplex`.

    Returns
    -------
    A, alpha, chi, phi : ndarray, shape (N,)
    """
    signal = _as_quaternion(signal)
    N = len(signal)
    A = qmod(signal)
    invA = np.divide(1., A, out=np.zeros_like(A), where=A != 0)
    sig = signal * invA[:, None]
    a, b, c, d = sig.T

    chi = 0.5 * np.arcsin(np.clip(2 * (b * c - a * d), -1, 1))
    alpha = np.zeros(N)
    phi_temp = np.zeros(N)
    spin = np.abs(chi) == np.pi / 4
    phi_temp[spin] = np.arctan2(2 * (b * d - a * c), a ** 2 - b ** 2 - c ** 2 + d ** 2)[spin] / 2
    ns = ~spin
    alpha[ns] = np.arctan2(2 * (a * b + c * d), a ** 2 - b ** 2 + c ** 2 - d ** 2)[ns] / 2
    phi_temp[ns] = np.arctan2(2 * (b * d + a * c), a ** 2 + b ** 2 - c ** 2 - d ** 2)[ns] / 2

    # remove the non-physical jumps of the orientation due to its restriction to [-pi/2, pi/2]
    for it in range(1, N):
        if abs(alpha[it] - alpha[it - 1]) > np.pi / 2:
            alpha_temp = alpha[it] - np.pi * np.sign(alpha[it] - alpha[it - 1])
            alpha[it] = np.mod(alpha_temp + np.pi, 2 * np.pi) - np.pi

    sig_temp = qprod(qexp(qprod(I_Q, alpha)), qexp(qprod(-chi, K_Q)), qexp(qprod(J_Q, phi_temp)))
    # control the sign of phi: the reconstruction must give back the signal (to 0.01), else phi is shifted by pi
    phi = np.zeros(N)
    same = qmod(sig_temp - sig) < 0.01
    opposite = ~same & (qmod(sig_temp + sig) < 0.01)
    phi[same] = phi_temp[same]
    phi[opposite] = np.where(phi_temp[opposite] >= 0, phi_temp[opposite] - np.pi, phi_temp[opposite] + np.pi)
    return A, alpha, chi, phi


def slow_flow_of_modes(eta_a, eta_b, dt, f0=Annular.omega / (2 * np.pi), bandwidth=50., t=None):
    r"""Band-pass both modes around `f0`, then return the slow flow ``(A, alpha, chi, phi)``.

    As `Fast_SDE_solv_lin_param_v2.m`: the modes are filtered with `bandpass_rect` in
    $[f_0 - B/2, f_0 + B/2]$, and, when `t` is given, the fast oscillation $2\pi f_0 t$ is removed
    from `phi`.
    """
    fs = 1. / dt
    ea = bandpass_rect(eta_a, f0 - bandwidth / 2, f0 + bandwidth / 2, fs)
    eb = bandpass_rect(eta_b, f0 - bandwidth / 2, f0 + bandwidth / 2, fs)
    A, alpha, chi, phi = slow_flow(analytic_hypercomplex(ea, eb))
    if t is not None:
        phi = phi - 2 * np.pi * f0 * np.asarray(t)
    return A, alpha, chi, phi
