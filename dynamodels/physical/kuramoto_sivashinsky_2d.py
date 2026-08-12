import matplotlib.pyplot as plt
import numpy as np

from ..integrator import DiscreteIntegrator
from ..model import Model


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

    | regime         | nu1 | nu2  | grid  | dt   |
    |----------------|-----|------|-------|------|
    | periodic       | 0.5 | 0.2  | 32x32 | 0.01 |
    | travelling     | 0.5 | 0.35 | 32x32 | 0.1  |
    | quasi-periodic | 0.5 | 0.1  | 32x32 | 0.01 |
    | chaotic        | 0.1 | 0.1  | 64x64 | 0.01 |
    | chaotic_B      | 0.3 | 0.1  | 64x64 | 0.01 |

    The default parameters are the 'chaotic' regime (leading Lyapunov exponent
    approximately 1.8).
    """

    t_transient = 100.
    t_CR = 10.

    Nq = 4                 # Number of sensors
    Nx = 64                # Grid points in x
    Ny = 64                # Grid points in y
    nu1 = 0.1              # Fourth-order 'viscosity' parameter
    nu2 = 0.1              # Anisotropic counterpart (alpha = nu2 / nu1)
    l = float(np.pi)       # noqa: E741 -- domain half-length (0, 2l]^2, matches the ks2d study notation

    Mcontour = 32          # Contour-integral points for the ETDRK4 coefficients
    Rcontour = 15.0        # Contour radius

    # structural parameters: carried by ntsa.respawn and encoded in Model.filename
    fixed_params = ['Nx', 'Ny', 'nu1', 'nu2', 'l']
    extra_print_params = ['Nx', 'Ny', 'nu1', 'nu2']
    sensor_placement_method = 'grid'

    def __init__(self, **model_dict):
        """Initialize the 2D KS model.

        Parameters
        ----------
        **model_dict
            Model parameters; supported keys are:

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
        self.Lhat = (self.kX**2 + self.aniso * self.kY**2
                     - self.nu1 * (self.kX**4
                                   + 2.0 * self.aniso * self.kX**2 * self.kY**2
                                   + self.aniso**2 * self.kY**4))

        self._etdrk4_cache = None

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

    @property
    def ETDRK4_f_terms(self):
        """ETDRK4 coefficient arrays (Nx, Ny), recomputed if dt changed."""
        if self._etdrk4_cache is None or self._etdrk4_cache[0] != self.dt:
            self._etdrk4_cache = (self.dt, self._compute_etdrk4_terms())
        return self._etdrk4_cache[1]

    def _compute_etdrk4_terms(self):
        """Kassam-Trefethen contour-integral evaluation of the ETDRK4
        coefficients for the diagonal spectral operator Lhat."""
        mm = np.arange(1, self.Mcontour + 1)
        r = self.Rcontour * np.exp(1j * np.pi * (mm - 0.5) / self.Mcontour)
        LR = self.dt * self.Lhat[:, :, None] + r[None, None, :]

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            terms = dict(
                E=np.exp(self.dt * self.Lhat),
                E2=np.exp(0.5 * self.dt * self.Lhat),
                Q=self.dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=2)),
                f1=self.dt * np.real(np.mean(
                    (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=2)),
                f2=self.dt * np.real(np.mean(
                    (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=2)),
                f3=self.dt * np.real(np.mean(
                    (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=2)),
            )
        return terms

    def _nonlinear_hat(self, vhat):
        """N(u) = -1/2 F[u_x^2 + alpha u_y^2] for spectral fields (Nx, Ny, m)."""
        vx = np.fft.ifft2(1j * self.kX[:, :, None] * vhat, axes=(0, 1)).real
        vy = np.fft.ifft2(1j * self.kY[:, :, None] * vhat, axes=(0, 1)).real
        return -0.5 * np.fft.fft2(vx * vx + self.aniso * vy * vy, axes=(0, 1))

    def ETDRK4_step(self, u):
        """One ETDRK4 step of physical fields u with shape (Nx, Ny, m)."""
        c = self.ETDRK4_f_terms
        E, E2, Q = c['E'][:, :, None], c['E2'][:, :, None], c['Q'][:, :, None]
        f1, f2, f3 = c['f1'][:, :, None], c['f2'][:, :, None], c['f3'][:, :, None]

        u = u - np.mean(u, axis=(0, 1), keepdims=True)
        vhat = np.fft.fft2(u, axes=(0, 1))

        Nv = self._nonlinear_hat(vhat)
        a = E2 * vhat + Q * Nv
        Na = self._nonlinear_hat(a)
        b = E2 * vhat + Q * Na
        Nb = self._nonlinear_hat(b)
        cstage = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = self._nonlinear_hat(cstage)

        vhat_next = E * vhat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        u_next = np.fft.ifft2(vhat_next, axes=(0, 1)).real
        return u_next - np.mean(u_next, axis=(0, 1), keepdims=True)

    def time_step(self, Nt=10, averaged=False, alpha=None):
        """Advance all m members Nt ETDRK4 steps.

        Returns
        -------
        psi : np.ndarray
            Trajectory of shape (Nt + 1, Nx*Ny, m), including the current state.
        t : np.ndarray
            Time points, shape (Nt + 1,).
        """
        u0 = self.current_state
        if u0.ndim == 1:
            u0 = u0[:, None]
        m = u0.shape[1]

        t = np.round(self.current_time + np.arange(Nt + 1) * self.dt, self.precision_t)

        if averaged and m > 1:
            u_mean = np.mean(u0, axis=1, keepdims=True)
            deviation = u0 - u_mean
            u = u_mean.reshape(self.Nx, self.Ny, 1)
            frames = [u]
            for _ in range(Nt):
                frames.append(self.ETDRK4_step(frames[-1]))
            psi = np.stack([f.reshape(-1, 1) + deviation for f in frames], axis=0)
        else:
            u = u0.reshape(self.Nx, self.Ny, m)
            frames = [u]
            for _ in range(Nt):
                frames.append(self.ETDRK4_step(frames[-1]))
            psi = np.stack([f.reshape(-1, m) for f in frames], axis=0)

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

    def visualize_spatiotemporal_hist(self, nframes=6, member=0, **kwargs):
        """Snapshot strip of the physical field u(x, y) for one ensemble member."""
        u_hist = self.get_observable_hist(loc='all')
        idx = np.linspace(0, u_hist.shape[0] - 1, nframes, dtype=int)
        lim = np.max(np.abs(u_hist[idx, :, member]))

        fig, axs = plt.subplots(ncols=nframes, figsize=(2.2 * nframes, 2.4),
                                sharey=True, constrained_layout=True)
        for ax, ii in zip(np.atleast_1d(axs), idx):
            im = ax.imshow(u_hist[ii, :, member].reshape(self.Nx, self.Ny).T,
                           origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim,
                           extent=[0, 2 * self.l, 0, 2 * self.l])
            ax.set(title=f"$t={self.hist_t[ii]:.2f}$", xlabel="$x$")
        np.atleast_1d(axs)[0].set(ylabel="$y$")
        fig.colorbar(im, ax=axs, shrink=0.8)
        fig.suptitle(rf"2D KS, $\nu_1={self.nu1}$, $\nu_2={self.nu2}$, "
                     rf"${self.Nx}\times{self.Ny}$")
