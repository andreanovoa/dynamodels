import matplotlib.pyplot as plt
import numpy as np

from ..integrator import DiscreteIntegrator
from ..model import Model

# Known regimes on the default l = pi domain (see the qlROM-DA study).
# Values not listed fall back to the class defaults (32x32 grid, dt = 0.01).
CASES = {
    'periodic': {'nu1': 0.5, 'nu2': 0.2, 'dt': 0.01},
    'travelling': {'nu1': 0.5, 'nu2': 0.35},
    'quasi-periodic': {'nu1': 0.5, 'nu2': 0.1},
    'chaotic': {'nu1': 0.1, 'nu2': 0.1, 'Nx': 64, 'Ny': 64,
                't_transient': 40., 't_CR': 4.},
    'chaotic_B': {'nu1': 0.3, 'nu2': 0.1, 'Nx': 64, 'Ny': 64},
}


class KS2D(Model):

    r"""Anisotropic two-dimensional Kuramoto-Sivashinsky equation.

    $$
    u_t + \left(u_{xx} + \alpha\, u_{yy}\right)
        + \nu_1 \left(\partial_x^2 + \alpha\, \partial_y^2\right)^2 u
        + \tfrac{1}{2}\left(u_x^2 + \alpha\, u_y^2\right) = 0,
    \qquad (x, y) \in (0, 2\ell]^2,
    $$

    doubly periodic, with anisotropy ratio $\alpha = \nu_2/\nu_1$. Solved
    pseudo-spectrally (full 2-D FFT) with the ETDRK4 scheme; the exponential
    coefficients are evaluated by the Kassam-Trefethen contour integral. The
    zero mode is projected out at every step (the equation only defines $u$ up
    to a constant), so the field stays zero-mean.

    The state is the PHYSICAL field flattened to shape (Nx*Ny, m) -- real
    valued, so observables are plain state rows and ensemble filters need no
    complex-state handling.

    Known regimes on the default $\ell = \pi$ domain (see the qlROM-DA study):

    | regime         | nu1 | nu2  | grid  | dt   | t_transient | t_CR |
    |----------------|-----|------|-------|------|-------------|------|
    | periodic       | 0.5 | 0.2  | 32x32 | 0.01 | 100.        | 10.  |
    | travelling     | 0.5 | 0.35 | 32x32 | 0.1  | 100.        | 10.  |
    | quasi-periodic | 0.5 | 0.1  | 32x32 | 0.01 | 100.        | 10.  |
    | chaotic        | 0.1 | 0.1  | 64x64 | 0.01 | 40.        | 4.  |
    | chaotic_B      | 0.3 | 0.1  | 64x64 | 0.01 | 100.        | 10.  |

    The class defaults are the 'periodic' regime; pass ``case='chaotic'`` etc.
    (see `CASES`) to select another. Explicit kwargs win over the case values,
    e.g. ``KS2D(case='chaotic', Nx=128, Ny=128)``.
    """

    t_transient = 100.
    t_CR = 10.

    case = None  # default case; can be overridden in __init__ or by kwargs
    Nq = 4                 # Number of sensors
    Nx = 32                # Grid points in x
    Ny = 32                # Grid points in y
    nu1 = 0.5              # Fourth-order 'viscosity' parameter
    nu2 = 0.2              # Anisotropic counterpart (alpha = nu2 / nu1)
    l = float(np.pi)       # noqa: E741 -- domain half-length (0, 2l]^2, matches the ks2d study notation

    Mcontour = 32          # Contour-integral points for the ETDRK4 coefficients
    Rcontour = 15.0        # Contour radius

    # nu1/nu2 are estimable (per-member ETDRK4 coefficients are rebuilt when a
    # DA analysis moves them); the grid and domain stay structural
    params = ['nu1', 'nu2']
    fixed_params = ['Nx', 'Ny', 'l']
    extra_print_params = ['Nx', 'Ny', 'nu1', 'nu2']
    sensor_placement_method = 'grid'

    def __init__(self, **model_dict):
        """Initialize the 2D KS model.

        Parameters
        ----------
        **model_dict
            Model parameters; supported keys are:

            - ``case`` : str, a named regime from `CASES`; its values are
              defaults that any explicit kwarg below overrides.
            - ``Nx``, ``Ny`` : int, grid points per direction (must be even).
            - ``nu1``, ``nu2`` : float, viscosity parameters (alpha = nu2/nu1).
            - ``l`` : float, domain half-length, domain is (0, 2l]^2.
            - ``dt`` : float, time step size.
            - ``Nq`` : int, number of sensors.
            - ``sensor_placement_method`` : str, ``'grid'`` or ``'random'``.
            - ``seed`` : int, random seed for sensor placement.
            - ``psi0`` : ndarray, initial PHYSICAL field, shape (Nx*Ny,) or
              (Nx*Ny, m) (optional; defaults to sin(X+Y) + sin(X) + sin(Y)).
        """
        self.case = model_dict.pop('case', None)
        if self.case is not None:
            if self.case not in CASES:
                raise ValueError(f"Unknown case '{self.case}'. Must be one of {list(CASES)}.")
            for key, val in CASES[self.case].items():
                model_dict.setdefault(key, val)  # explicit kwargs win over the case

        for key in list(model_dict.keys()):
            if key in vars(KS2D):
                setattr(self, key, model_dict.pop(key))

        if self.Nx % 2 != 0 or self.Ny % 2 != 0:
            raise ValueError("Nx and Ny must be even.")
        if self.nu1 <= 0 or self.nu2 <= 0:
            raise ValueError("nu1 and nu2 must be positive.")

        # Physical grid on (0, 2l]^2
        self.dx = 2.0 * self.l / self.Nx
        self.dy = 2.0 * self.l / self.Ny
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Full FFT wavenumbers (positive Nyquist convention)
        dk = np.pi / self.l
        kx = np.concatenate((np.arange(0, self.Nx // 2 + 1),
                             np.arange(-self.Nx // 2 + 1, 0))) * dk
        ky = np.concatenate((np.arange(0, self.Ny // 2 + 1),
                             np.arange(-self.Ny // 2 + 1, 0))) * dk
        self.kX, self.kY = np.meshgrid(kx, ky, indexing='ij')

        self.aniso = self.nu2 / self.nu1     # alpha in the equation; Model reserves .alpha
        self.Lhat = self._lhat(self.nu1, self.nu2)

        self._etdrk4_cache = None
        self._member_terms = None

        #  Select sensors ___________________________ #
        if self.sensor_placement_method == 'grid':
            self.sensor_locations = np.linspace(0, self.Nx * self.Ny - 1, self.Nq,
                                                endpoint=True, dtype=int)
        elif self.sensor_placement_method == 'random':
            self.sensor_locations = self.rng.integers(0, self.Nx * self.Ny - 1, self.Nq)
        else:
            raise NotImplementedError(f"sensor_placement_method "
                                      f"'{self.sensor_placement_method}' not recognized.")

        #   Init Model  #
        dt = model_dict.pop('dt', 0.01)
        psi0 = model_dict.pop('psi0', None)
        if psi0 is None:
            u0 = np.sin(self.X + self.Y) + np.sin(self.X) + np.sin(self.Y)
            u0 -= np.mean(u0)
            psi0 = u0.reshape(-1, 1)

        super().__init__(psi0=psi0, dt=dt, integrator_class=DiscreteIntegrator, **model_dict)

        self.alpha_labels = dict(nu1='$\\nu_1$', nu2='$\\nu_2$')
        self.alpha_lims = dict(nu1=(0.01, 1.0), nu2=(0.01, 1.0))

    # _______________ Modified Model methods ________________ #

    @property
    def obs_labels(self):
        return [f"$u(\\mathbf{{x}}_{{{j+1}}})$" for j in np.arange(self.Nq)]

    @property
    def state_labels(self):
        return [f"$u_{{{j+1}}}$" for j in np.arange(self.Nphi)]

    def get_observables(self, Nt=1, loc=None, **kwargs):
        """Observable state at the sensor locations (the state is already the
        physical field, so observables are plain rows of the flattened state).

        Parameters
        ----------
        Nt : int
            Number of time steps to retrieve. Default is 1.
        loc : array-like or str, optional
            Flattened-grid sensor indices; 'all' returns the whole field;
            None (default) uses the model's sensor locations.
        """
        if loc is None:
            loc = self.sensor_locations
        elif isinstance(loc, str) and loc.lower() == 'all':
            loc = np.arange(self.Nx * self.Ny)
        elif isinstance(loc, str):
            raise ValueError("loc must be None, 'all', or array-like integer indices.")

        loc = np.asarray(loc, dtype=int)

        if Nt == 1:
            return self.hist[-1, loc]
        else:
            return self.hist[-Nt:, loc]

    # _______________ KS2D specific properties and methods ________________ #

    @property
    def x(self):
        return np.arange(self.Nx) * self.dx

    @property
    def y(self):
        return np.arange(self.Ny) * self.dy

    def _lhat(self, nu1, nu2):
        """Diagonal spectral operator for given nu values. Scalars give the
        shared (Nx, Ny) operator; per-member arrays of length m give a
        (Nx, Ny, m) stack."""
        nu1, nu2 = np.asarray(nu1, dtype=float), np.asarray(nu2, dtype=float)
        kX, kY = (self.kX, self.kY) if nu1.ndim == 0 else (self.kX[:, :, None], self.kY[:, :, None])
        a = nu2 / nu1
        return (kX**2 + a * kY**2
                - nu1 * (kX**4 + 2.0 * a * kX**2 * kY**2 + a**2 * kY**4))

    @property
    def ETDRK4_f_terms(self):
        """ETDRK4 coefficient arrays (Nx, Ny) for the constructed (nu1, nu2),
        recomputed if dt changed."""
        if self._etdrk4_cache is None or self._etdrk4_cache[0] != self.dt:
            self._etdrk4_cache = (self.dt, self._compute_etdrk4_terms())
        return self._etdrk4_cache[1]

    def _compute_etdrk4_terms(self, Lhat=None):
        """Kassam-Trefethen contour-integral evaluation of the ETDRK4
        coefficients for the diagonal spectral operator `Lhat` (defaults to
        the constructed one; a (Nx, Ny, m) stack gives per-member terms)."""
        if Lhat is None:
            Lhat = self.Lhat
        mm = np.arange(1, self.Mcontour + 1)
        r = self.Rcontour * np.exp(1j * np.pi * (mm - 0.5) / self.Mcontour)
        LR = self.dt * Lhat[..., None] + r

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            terms = dict(
                E=np.exp(self.dt * Lhat),
                E2=np.exp(0.5 * self.dt * Lhat),
                Q=self.dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=-1)),
                f1=self.dt * np.real(np.mean(
                    (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=-1)),
                f2=self.dt * np.real(np.mean(
                    (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=-1)),
                f3=self.dt * np.real(np.mean(
                    (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=-1)),
            )
        return terms

    def _terms_for(self, alpha):
        """ETDRK4 terms (plus per-member anisotropy) for a list of per-member
        alpha dicts; falls through to the shared terms when every member sits
        at the constructed (nu1, nu2)."""
        nu1 = np.array([a['nu1'] for a in alpha], dtype=float)
        nu2 = np.array([a['nu2'] for a in alpha], dtype=float)
        if np.all(nu1 == self.nu1) and np.all(nu2 == self.nu2):
            return self.ETDRK4_f_terms
        key = (self.dt, nu1.tobytes(), nu2.tobytes())
        if self._member_terms is None or self._member_terms[0] != key:
            # ponytail: full contour rebuild whenever the nus move -- cheap next
            # to stepping, and only a DA analysis (or a new alpha) moves them
            terms = self._compute_etdrk4_terms(self._lhat(nu1, nu2))
            terms['aniso'] = nu2 / nu1
            self._member_terms = (key, terms)
        return self._member_terms[1]

    def _nonlinear_hat(self, vhat, aniso=None):
        """N(u) = -1/2 F[u_x^2 + alpha u_y^2] for spectral fields (Nx, Ny, m);
        `aniso` may be scalar or per-member (m,)."""
        if aniso is None:
            aniso = self.aniso
        vx = np.fft.ifft2(1j * self.kX[:, :, None] * vhat, axes=(0, 1)).real
        vy = np.fft.ifft2(1j * self.kY[:, :, None] * vhat, axes=(0, 1)).real
        return -0.5 * np.fft.fft2(vx * vx + aniso * vy * vy, axes=(0, 1))

    def ETDRK4_step(self, u, terms=None):
        """One ETDRK4 step of physical fields u with shape (Nx, Ny, m).
        `terms` defaults to the shared coefficients; per-member (Nx, Ny, m)
        stacks from `_terms_for` are used as-is."""
        c = self.ETDRK4_f_terms if terms is None else terms
        E, E2, Q, f1, f2, f3 = (c[k][:, :, None] if c[k].ndim == 2 else c[k]
                                for k in ('E', 'E2', 'Q', 'f1', 'f2', 'f3'))
        aniso = c.get('aniso', self.aniso)

        u = u - np.mean(u, axis=(0, 1), keepdims=True)
        vhat = np.fft.fft2(u, axes=(0, 1))

        Nv = self._nonlinear_hat(vhat, aniso)
        a = E2 * vhat + Q * Nv
        Na = self._nonlinear_hat(a, aniso)
        b = E2 * vhat + Q * Na
        Nb = self._nonlinear_hat(b, aniso)
        cstage = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = self._nonlinear_hat(cstage, aniso)

        vhat_next = E * vhat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        u_next = np.fft.ifft2(vhat_next, axes=(0, 1)).real
        return u_next - np.mean(u_next, axis=(0, 1), keepdims=True)

    def time_step(self, Nt=10, averaged=False, alpha=None):
        """Advance all m members Nt ETDRK4 steps.

        Parameters
        ----------
        alpha : list of dict, optional
            Per-member parameter dicts (as from `get_alpha`); defaults to the
            values carried in the (possibly augmented) current state. Members
            with nus away from the constructed ones get their own ETDRK4
            coefficients.

        Returns
        -------
        psi : np.ndarray
            Trajectory of shape (Nt + 1, Nphi [+ Na], m), including the
            current state; estimated-parameter rows are carried unchanged.
        t : np.ndarray
            Time points, shape (Nt + 1,).
        """
        psi0 = self.current_state
        if psi0.ndim == 1:
            psi0 = psi0[:, None]
        m = psi0.shape[1]
        Nu = self.Nx * self.Ny
        u0, aug = psi0[:Nu], psi0[Nu:]   # estimated-parameter rows ride along unchanged

        if alpha is None:
            alpha = self.get_alpha(psi0)

        t = np.round(self.current_time + np.arange(Nt + 1) * self.dt, self.precision_t)

        if averaged and m > 1:
            mean_alpha = [{k: float(np.mean([a[k] for a in alpha])) for k in self.params}]
            terms = self._terms_for(mean_alpha)
            u_mean = np.mean(u0, axis=1, keepdims=True)
            deviation = u0 - u_mean
            u = u_mean.reshape(self.Nx, self.Ny, 1)
            frames = [u]
            for _ in range(Nt):
                frames.append(self.ETDRK4_step(frames[-1], terms))
            psi = np.stack([f.reshape(-1, 1) + deviation for f in frames], axis=0)
        else:
            terms = self._terms_for(alpha)
            u = u0.reshape(self.Nx, self.Ny, m)
            frames = [u]
            for _ in range(Nt):
                frames.append(self.ETDRK4_step(frames[-1], terms))
            psi = np.stack([f.reshape(-1, m) for f in frames], axis=0)

        if aug.shape[0]:
            psi = np.concatenate([psi, np.broadcast_to(aug, (Nt + 1, *aug.shape))], axis=1)
        return psi, t

    def get_energy(self, Nt=0, u=None):
        """Spatially averaged L2 energy, E = mean(u^2), shape (Nt, m)."""
        if u is None:
            u = self.get_observable_hist(Nt=Nt, loc='all')
        if u.ndim == 2:
            u = u[np.newaxis, :]
        assert u.shape[1] == self.Nx * self.Ny
        return np.mean(u**2, axis=1)

    def get_enstrophy(self, Nt=0, u=None):
        """Spatially averaged enstrophy, mean(u_x^2 + alpha u_y^2), shape (Nt, m)."""
        if u is None:
            u = self.get_observable_hist(Nt=Nt, loc='all')
        if u.ndim == 2:
            u = u[np.newaxis, :]
        Nt_, _, m = u.shape
        vhat = np.fft.fft2(u.reshape(Nt_, self.Nx, self.Ny, m), axes=(1, 2))
        ux = np.fft.ifft2(1j * self.kX[None, :, :, None] * vhat, axes=(1, 2)).real
        uy = np.fft.ifft2(1j * self.kY[None, :, :, None] * vhat, axes=(1, 2)).real
        return np.mean((ux**2 + self.aniso * uy**2).reshape(Nt_, -1, m), axis=1)

    def visualize_spatiotemporal_hist(self,
                                      nframes=6, member=0, **kwargs):
        """Snapshot strip of the physical field u(x, y) for one ensemble member,
        plus a space-time diagram of the mid-domain slice u(x, y = l, t)."""
        u_hist = self.get_observable_hist(loc='all')
        t = self.hist_t
        u = u_hist[:, :, member].reshape(len(t), self.Nx, self.Ny)
        idx = np.linspace(0, len(t) - 1, nframes, dtype=int)
        lim = np.max(np.abs(u[idx]))

        fig = plt.figure(figsize=(2.2 * nframes, 4.8), constrained_layout=True)
        gs = fig.add_gridspec(2, nframes)
        axs = [fig.add_subplot(gs[0, jj]) for jj in range(nframes)]
        im = None
        for ax, ii in zip(axs, idx):
            im = ax.imshow(u[ii].T,
                           origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim,
                           extent=(0, 2 * self.l, 0, 2 * self.l))
            ax.set(title=f"$t={t[ii]:.2f}$", xlabel="$x$")
        axs[0].set(ylabel="$y$")
        assert im is not None
        fig.colorbar(im, ax=axs, shrink=0.8)

        # space-time diagram along the mid-domain slice y = l
        u_slice = u[:, :, self.Ny // 2]
        ax_st = fig.add_subplot(gs[1, :])
        lim_st = np.max(np.abs(u_slice))
        im_st = ax_st.imshow(u_slice.T, aspect='auto', origin='lower',
                             cmap='RdBu_r', vmin=-lim_st, vmax=lim_st,
                             extent=(t[0], t[-1], 0, 2 * self.l))
        ax_st.set(xlabel="$t$", ylabel="$x$", title=r"$u(x, y = l, t)$")
        fig.colorbar(im_st, ax=ax_st, shrink=0.8)

        fig.suptitle(rf"2D KS, $\nu_1={self.nu1}$, $\nu_2={self.nu2}$, "
                     rf"${self.Nx}\times{self.Ny}$")
