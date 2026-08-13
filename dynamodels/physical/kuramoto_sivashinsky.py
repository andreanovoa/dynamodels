

# %%


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as cm
from matplotlib.colors import Normalize

from ..integrator import DiscreteIntegrator
from ..model import Model


class KS(Model):

    r"""Kuramoto-Sivashinsky equation.

    $$
    u_t + u_{xx} + \nu\, u_{xxxx} + u\,u_x = 0, \qquad x \in (0, L],
    $$

    with periodic boundary conditions $u(t, 0) = u(t, L)$, $u_x(t, 0) = u_x(t, L)$.
    Solved with the ETDRK4 scheme in Fourier space, where the Fourier transform pair is

    $$
    \hat{u}(k) = \mathcal{F}[u(x)] = \frac{1}{L}\int_0^L u(x)\,e^{-ikx}\,\mathrm{d}x,
    \qquad
    u(x) = \mathcal{F}^{-1}[\hat{u}(k)] = \sum_k \hat{u}(k)\,e^{ikx}.
    $$

    On the $N_x$-point grid the wavenumbers are $\alpha_j = 2\pi j / L$, so the
    (diagonal) linear operator is $\alpha_j^2 - \nu\,\alpha_j^4$ and the nonlinear
    term $u\,u_x$ is computed in physical space and transformed back to Fourier
    space at every stage.

    **Parametrization ($\nu$, $L$).** The pair is independent: whichever of the two
    is given fixes that side of the operator.

    - `nu` only (the default): the domain follows the standard nondimensionalization
      $L = 2\pi/\sqrt{\nu}$, and the equation is then integrated in its $\nu = 1$
      form on that domain -- so `self.nu` is **1** afterwards and the stored
      $(N_x, \nu, L)$ always describes the operator that was actually integrated
      (this is what makes `ntsa.respawn`, which rebuilds from `fixed_params`,
      bit-faithful).
    - `L` only: $\nu = 1$ on the given domain.
    - **both**: both are honoured as given, i.e. the genuine two-parameter system
      $u_t + u_{xx} + \nu u_{xxxx} + u u_x = 0$ on $(0, L]$.

    The one- and two-parameter forms are related by $v(x', t') = \sqrt{\nu}\,u(x, t)$
    with $x = \sqrt{\nu}\,x'$ and $t = \nu\,t'$: `KS(Nx, L=Lx, nu=visc, dt=dt)` and
    `KS(Nx, L=Lx/sqrt(visc), dt=dt/visc)` describe the same physical system.
    """

    # name: str = 'KS'
    t_transient = 300.
    t_CR = 50.

    Nq = 4               # Number of sensors
    Nx = 256             # Spatial discretization
    nu = 0.08            # 'Viscosity' parameter of the KS equation.
    L = -1         # Domain length (0, L] NB. Will be set to 2*pi/sqrt(nu) if not specified.
    # (nu, L) resolution rules: see the class docstring. Passing BOTH keeps both.

    initial_amplitude = 0.01

    # structural constructor params: carried by ntsa.respawn and encoded in filename
    fixed_params = ['Nx', 'nu', 'L']
    extra_print_params = ['Nx']
    sensor_placement_method = 'grid'


    def __init__(self, **model_dict):
        """Initialize the KS model.

        Sets up the spatial grid, wavenumbers, sensor locations, and initial state.

        Parameters
        ----------
        **model_dict
            Model parameters; supported keys are:

            - ``Nx`` : int, number of spatial grid points (must be even).
            - ``nu`` : float, viscosity parameter.
            - ``L`` : float, domain length, domain is (0, L].
              ``nu`` and ``L`` are independent -- see the class docstring for the
              resolution rules when only one of them is given.
            - ``dt`` : float, time step size.
            - ``initial_amplitude`` : float, amplitude of the initial condition.
            - ``Nq`` : int, number of sensors.
            - ``sensor_placement_method`` : str, ``'grid'`` or ``'random'``.
            - ``seed`` : int, random seed for sensor placement.
            - ``psi0`` : ndarray, initial state in Fourier space (optional).
        """


        # 'nu' has a non-sentinel class default, so an EXPLICIT nu is what marks the
        # two-parameter form; 'L' uses its non-positive class default as the sentinel.
        nu_given = model_dict.get('nu') is not None

        for key in list(model_dict.keys()):
            if key in vars(KS):
                setattr(self, key, model_dict.pop(key))


        if self.Nx % 2 != 0:
            raise ValueError("Nx must be even.")

        L_given = self.L is not None and self.L > 0

        if not L_given and not nu_given and self.nu is None:
            raise ValueError("Either L or nu must be specified.")
        elif not L_given:
            # nu alone: standard nondimensionalization. The domain absorbs nu and the
            # equation is integrated in its nu = 1 form, so (Nx, nu, L) stays a faithful
            # description of the operator (respawn / filename keying).
            self.L = 2 * np.pi / np.sqrt(self.nu)
            self.nu = 1.
        elif not nu_given:
            self.nu = 1.
        # else: both given -- honour both (general two-parameter form).

        assert self.L is not None and self.L > 0, "L must be positive."
        assert self.nu is not None and self.nu > 0, "nu must be positive."

        # Fourier wavenumbers alpha_j = 2 pi j / L on the domain (0, L]
        self.k = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.L / self.Nx)

        dt_requested = model_dict.pop('dt', 0.25)
        self.dt = dt_requested


        self.ETDRK4_f_terms = None  # This simply trigers the setter method.


        #  Select sensors ___________________________ #
        if self.sensor_placement_method not in ['grid', 'random']:
            raise NotImplementedError(f"sensor_placement_method '{self.sensor_placement_method}' not recognized.")

        if self.sensor_placement_method == 'grid':
            # Place sensors evenly spaced across the domain
            self.sensor_locations = np.linspace(0, self.Nx-1, self.Nq, endpoint=True, dtype=int)
        elif self.sensor_placement_method == 'random':
            # Place sensors at random locations in the domain
            self.sensor_locations = self.rng.integers(0, self.Nx-1, self.Nq)


        #   Init Model  #
        psi0 = model_dict.pop('psi0', None)
        if psi0 is None:
            # Initialize state in physical space and transform to spectral space
            u0 = self.initial_amplitude * self.rng.standard_normal(self.Nx)
            u0 -= np.mean(u0)  # Zero-mean initial condition
            u_hat = KS.physical_to_fourier(u0)[:, None]     # Transform to Fourier space
            psi0 = np.array(u_hat)


        super().__init__(psi0=psi0, dt=self.dt, integrator_class=DiscreteIntegrator, **model_dict)

        # Model's dt setter rounds to precision_t decimals, which silently perturbs
        # timesteps with more significant digits (e.g. dt = 0.1 * 71 / 16). Keep the
        # exact requested value for stepping (precision_t still governs time stamps)
        # and rebuild the ETDRK4 coefficients with it.
        self._dt = float(dt_requested)
        self.ETDRK4_f_terms = None


    # _______________ Modified Model methods ________________ #


    @property
    def obs_labels(self):
        return [f"$u(x_{{{j+1}}})$" for j in np.arange(self.Nq)]


    @property
    def state_labels(self):
        return [f"$\\hat{{u}}_{{{j+1}}}$" for j in np.arange(self.Nphi)]


    def get_observables(self, Nt=1, loc=None, **kwargs):
        """
        Get the observable state in physical space at specified sensor locations.
        Parameters
        ----------
        Nt : int
            Number of time steps to retrieve. Default is 1.
        loc : array-like or str, optional
            Sensor locations to retrieve observables from. If 'all', returns observables at all spatial points.
            If None, returns observables at the predefined sensor locations.
        """
        if loc is None:
            loc = self.sensor_locations
        elif loc.lower() == 'all':
            loc = np.arange(self.Nx)

        if Nt == 1:
            return KS.fourier_to_physical(self.hist[-1, :self.Nk])[loc]
        else:
            return KS.fourier_to_physical(self.hist[-Nt:, :self.Nk])[:, loc]


    # _______________ KS specific properties and methods ________________ #

    @property
    def x(self):
        return  np.arange(self.Nx) *  self.L / self.Nx


    @property
    def Nk(self):
        return self.k.shape[0]


    def first_derivative_x(self, u_hat):
        if u_hat.ndim == 2:
            assert u_hat.shape[0] == self.Nk, f'u_hat.shape[0] == {u_hat.shape[0]} != {self.Nk}'
            return 1.j * self.k[:, None] * u_hat
        elif u_hat.ndim == 3:

            assert u_hat.shape[1] == self.Nk, f'u_hat.shape[1] == {u_hat.shape[1]} != {self.Nk}'
            return 1.j * self.k[None, :, None] * u_hat
        else:
            raise AssertionError(f'u_hat should be shape (Nk, m)=({self.Nk, self.m}) or (Nt, Nk, m). Got {u_hat.shape} instead.')


    def __nonlinear_operator(self, u_hat):
        """
            Compute the nonlinear term N(u) = -u * u_x in Fourier space.
            F[-u * u_x] = F[1/2(u * u)_x]
            Input: (N_x, m)
            # rfft outputs the positive frequencies n/2+1 if even, (n+1)/2 if odds
        """
        assert u_hat.shape[0] == self.Nk, f'u_hat.shape[0] == {u_hat.shape[0]} != {self.Nk}'


        # Dealias using 2/3 rule
        cutoff = int(self.Nx * 2/3)
        dealias = np.ones_like(self.k, dtype=bool)
        dealias[cutoff:-cutoff] = False

        # Apply filter to input
        u_hat_filtered = u_hat.copy()
        u_hat_filtered[~dealias] = 0.0

        # Option A-----
        # Square in thw physical space and transform back
        u = KS.fourier_to_physical(u_hat_filtered)
        u2_hat = KS.physical_to_fourier(u**2)

        N_hat =  - 0.5 * self.first_derivative_x(u2_hat)
        #-----  Option A


        # # Option B-----(numerically equivalenrt. A is faster.)
        # # Transform to physical space
        # u = KS.fourier_to_physical(u_hat_filtered)

        # # Compute derivative explicitly
        # # u_x = np.real(ifft(1j * self.k * u_hat_filtered))
        # u_x = KS.fourier_to_physical(self.first_derivative_x(u_hat_filtered) )
        # nonlinear = -u * u_x

        # # Transform back to Fourier space and apply filter
        # N_hat = KS.physical_to_fourier(nonlinear)
        # #-----  Option B

        N_hat[~dealias] = 0.0

        return N_hat

    @property
    def __linear_operator(self):
        # Fourier multipliers of the linear term: alpha^2 - nu alpha^4, with
        # alpha = self.k = 2 pi j / L. The single-parameter forms resolve nu to 1
        # in __init__, so this reduces to alpha^2 - alpha^4 for them.
        return (self.k**2 - self.nu * self.k**4)[:, None]


    @property
    def ETDRK4_f_terms(self):
        return self._ETDRK4_f_terms



    @ETDRK4_f_terms.setter
    def ETDRK4_f_terms(self, _):
        """Kassam-Trefethen ETDRK4 coefficients (E, E2, Q, f1, f2, f3) per Fourier mode.

        The exponentials are exact; the phi-function coefficients Q/f1/f2/f3 are
        evaluated with the resolvent contour integral on the half circle |z| = 15
        (M = 32 points, real-part symmetrization) around the origin -- the same
        quadrature rule used for the dense-operator ETDRK4 coefficients in the
        qlROM stack, so a Galerkin projection of this discretization reproduces
        this model's discrete map. Note: for modes with |dt*L| > 15 (far outside
        the contour) the quadrature returns ~0 instead of the tiny exact value;
        those modes are overdamped (E ~ exp(dt*L) ~ 0) and their nonlinear-response
        error is negligible, matching the established dense-operator behavior.
        """
        L = self.__linear_operator

        M = 32          # contour quadrature points
        R = 15.0        # contour radius
        z = R * np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)   # (M,)
        ez, ez2 = np.exp(z), np.exp(z / 2)

        # resolvent 1/(z - dt*L) per (mode, contour point)
        res = 1.0 / (z[np.newaxis, :] - self.dt * L)                    # (Nk, 1) -> (Nk, M)

        hQ = ez2 - 1.0
        hf1 = (-4.0 - z + ez * (4.0 - 3.0 * z + z**2)) / z**2
        hf2 = (2.0 + z + ez * (z - 2.0)) / z**2
        hf3 = (-4.0 - 3.0 * z - z**2 + ez * (4.0 - z)) / z**2

        def _coef(h):
            return self.dt * np.real(np.mean(h[np.newaxis, :] * res, axis=-1))[:, None]

        self._ETDRK4_f_terms = dict(
            E=np.exp(self.dt * L),
            E2=np.exp(self.dt * L / 2),
            Q=_coef(hQ),
            f1=_coef(hf1),
            f2=_coef(hf2),
            f3=_coef(hf3),
            nonlinear_operator=self.__nonlinear_operator,
        )



    @staticmethod
    def ETDRK4_step(u_hat, nonlinear_operator, E, E2, Q, f1, f2, f3):
        """
        Standard Kassam-Trefethen ETDRK4 step:

        a_n = exp(L h / 2) u_n + Q N(u_n)
        b_n = exp(L h / 2) u_n + Q N(a_n)
        c_n = exp(L h / 2) a_n + Q (2 N(b_n) - N(u_n))

        u_{n+1} = exp(L h) u_n + f1 N(u_n) + 2 f2 (N(a_n) + N(b_n)) + f3 N(c_n)

        where h is the timestep, L the (diagonal) linear operator, N the
        nonlinear operator, and Q, f1, f2, f3 the contour-integrated
        phi-function coefficients (see ETDRK4_f_terms). The linear part is
        integrated exactly; the nonlinear terms with fourth-order accuracy.
        """

        N1 = nonlinear_operator(u_hat)
        a = E2 * u_hat + Q * N1
        N2 = nonlinear_operator(a)
        b = E2 * u_hat + Q * N2
        N3 = nonlinear_operator(b)
        c = E2 * a + Q * (2 * N3 - N1)
        N4 = nonlinear_operator(c)

        return E * u_hat + f1 * N1 + 2 * f2 * (N2 + N3) + f3 * N4




    def time_step(self, Nt=10, averaged=False, alpha=None):
        """
        Integrator for the KS model that supports ensembles and averaged ensemble propagation.
        Matches interface conventions of other models.

        Parameters
        ----------
        Nt : int
            Number of time steps to integrate.
        averaged : bool, optional
            If True, integrates the mean state and broadcasts ensemble deviations.
        alpha : optional
            Additional model parameters.

        Returns
        -------
        psi : np.ndarray
            Forecasted state array of shape (Nt, Nphi, m).
        t : np.ndarray
            Time vector corresponding to each forecasted state.
        """

        u0_hat = self.current_state

        if u0_hat.ndim == 1:  # reshape for non-ensemble
            u0_hat = u0_hat[:, None]

        t = np.round(self.current_time + np.arange(Nt + 1) * self.dt, self.precision_t)


        if averaged and self.ensemble:
            u0_hat_mean = np.mean(u0_hat, axis=1, keepdims=True)
            psi_deviation = u0_hat - u0_hat_mean

            psi_mean_arr = [u0_hat_mean[:, 0]]
            for _ in range(Nt):
                psi_mean_arr.append(KS.ETDRK4_step(psi_mean_arr[-1][:, None], **self.ETDRK4_f_terms)[:, 0])
            psi_mean_arr = np.stack(psi_mean_arr, axis=0)  # (Nt+1, N_x)

            # Broadcast deviations
            psi = np.array([psi_mean_arr[ii][:, None] + psi_deviation for ii in range(psi_mean_arr.shape[0])])  # (Nt+1, N_x, m)

        else:
            # Single member integration
            psi = [u0_hat]
            for _ in range(Nt):
                psi.append(KS.ETDRK4_step(psi[-1], **self.ETDRK4_f_terms))

            psi = np.stack(psi, axis=0)


        return psi, t



    def get_energy(self, Nt=0, u=None):
        """
        Compute the L2 energy of the solution: E = (1/L) * integral(u^2)dx

        Returns:
        --------
        float
            L2 energy
        """

        if u is None:
            u = self.get_observable_hist(Nt=Nt, loc="all")


        if u.ndim == 2:
            u = u[np.newaxis, :]

        assert u.shape[1] == self.Nx

        return np.mean(u**2, axis=1)



    def get_enstrophy(self, Nt=0, u_hat=None):
        """
        Compute the enstrophy (integral of (u_x)^2).

        Returns:
        --------
        float
            Enstrophy
        """

        if u_hat is None:
            if Nt != 1:
                u_hat = self.hist[-Nt:]
            else:
                u_hat = self.current_state[np.newaxis, :]
        else:
            if u_hat.ndim == 2:
                u_hat = u_hat[np.newaxis, :]

        assert u_hat.shape[1] == self.k.shape[0]

        u_x_hat = self.first_derivative_x(u_hat)
        u_x = self.fourier_to_physical(u_x_hat)

        return np.mean(u_x**2, axis=1)


    @staticmethod
    def fourier_to_physical(u_hat):
        # Inverse real FFT
        if u_hat.ndim > 2:
            ax = 1
        else:
            ax = 0
        return np.fft.irfft(u_hat, axis=ax, n=None)

    @staticmethod
    def physical_to_fourier(u):
        # Real-to-complex FFT along spatial dimension
        if u.ndim > 2:
            ax = 1
        else:
            ax = 0
        return np.fft.rfft(u, axis=ax)



    def visualize_spatiotemporal_hist(self, y_hist=None, t=None, nrows=None, averaged=False, **kwargs):
        """
        Visualize the spatiotemporal evolution of the KS model in the physical space.
        """

        if y_hist is None:
            y_hist = self.get_observable_hist(loc="all")

        if t is None:
            t = self.hist_t

        if not averaged:
            if nrows is None:
                nrows = min(10, y_hist.shape[-1])

            fig = plt.figure(figsize=(10, 1.5 * nrows))
            axs = fig.subplots(nrows=nrows, sharey=True, sharex=True)
            if nrows == 1:
                axs = [axs]

            lim = np.max(abs(y_hist))

            for mi, ax in enumerate(axs):
                im = ax.imshow(y_hist[:, :, mi].T,
                            aspect='auto', origin='lower',
                            cmap='RdBu_r', vmin=-lim, vmax=lim,
                            extent=[t[0], t[-1], self.x[0], self.x[-1]])  # TRANSPOSE


            axs[0].set(title=rf"KS spatiotemporal evolution. $L={self.L/np.pi:.2f}\pi, \nu={self.nu}$")
            axs[-1].set(xlabel="$t$")

            fig.colorbar(im, ax=axs, orientation='vertical', shrink=1/nrows)  #type: ignore
        else:
            # Averaged ensemble visualization
            y_mean_hist = np.mean(y_hist, axis=-1)

            fig, axs = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)

            # Mean evolution
            lim_mean = np.max(abs(y_mean_hist))
            im0 = axs[0].imshow(y_mean_hist.T,
                                aspect='auto', origin='lower',
                                cmap='RdBu_r', vmin=-lim_mean, vmax=lim_mean,
                                extent=[t[0], t[-1], self.x[0], self.x[-1]])
            axs[0].set(title=rf"KS averaged spatiotemporal evolution (mean and std). $L={self.L/np.pi:.2f}\pi, \nu={self.nu}$") #type: ignore
            fig.colorbar(im0, ax=axs[0], orientation='vertical')

            # Deviation covariance evolution

            var_ensemble = np.var(y_hist, axis=-1, ddof=1).T            # (Nt, Nx)
            var_ensemble = np.sqrt(var_ensemble)                     # Standard deviation

            lim_dev = np.max(abs(var_ensemble))
            im1 = axs[1].imshow(var_ensemble,  # Plot covariance of deviations
                                aspect='auto', origin='lower',
                                cmap='magma', vmin=0, vmax=lim_dev,
                                extent=[t[0], t[-1], self.x[0], self.x[-1]])

            fig.colorbar(im1, ax=axs[1], orientation='vertical')
        # add the ticks and labels

        # Set spatial ticks as multiples of L
        assert self.L is not None, "L must be defined to set spatial ticks."
        ticks = (np.arange(4) + 1)* self.L/4
        tick_labels = [r"$L/4$", r"$L/2$", r"$3L/4$",r"$L$"]
        for ax in axs:
            ax.set(ylabel="$x$", yticks=ticks, yticklabels=tick_labels)


    @staticmethod
    def plot_temporal_E(model, Nt=0, max_lines=10):

        energy = model.get_energy(Nt=Nt)
        enstrophy = model.get_enstrophy(Nt=Nt)

        max_lines = min(model.m, max_lines)
        plot_m = np.arange(max_lines)


        c_energy = plt.get_cmap('tab20b', max_lines)
        c_enstrophy = plt.get_cmap('tab20b', max_lines)


        _, axs = plt.subplots(ncols=1, nrows=2)
        for ax, E, ttl, cmap in zip(axs, [energy, enstrophy], ['Energy', 'Enstrophy'], [c_energy, c_enstrophy]):
            for mi in plot_m:
                ax.plot(model.hist_t, E[:, mi], c=cmap(mi / max_lines))

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max_lines-1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax,
                                orientation='vertical', ticks=plot_m)
            cbar.ax.set_yticklabels([str(t) for t in plot_m])
            ax.set_ylabel(ttl)
            cbar.set_label('Member index', fontsize=12)

        axs[-1].set_xlabel("t")



# %%


if __name__ == "__main__":

    nu = .08
    dt = 0.25
    Nt = int(100 / dt)
    Nx = 256
    # L = 36

    import time

    t1 = time.time()

    for seed in [0]:


        model = KS( Nx=Nx,
                    dt=dt,
                    nu=nu,
                    seed=seed,
                    initial_amplitude=1.)

        # model.init_ensemble(std_psi=0.1,
        #                             m=10)


        print(f"Domain size: L = {model.L  }")
        print(f"Grid points: N = {model.Nx}")
        print(f"Time step: dt = {model.dt:.6f}")
        print(f"Viscosity: nu = {model.nu:.6f}")

        print(len(model.k))

        print(f"Domain size: L = {model.L  }")
        print(f"Grid points: N = {model.Nx}")
        print(f"Time step: dt = {model.dt:.6f}")
        print(f"Viscosity: nu = {model.nu:.6f}")



        solution, times = model.time_integrate(Nt=Nt)
        model.update_history(psi=solution, t=times)

        KS.visualize_spatiotemporal_hist(model)
        KS.plot_temporal_E(model=model)



    print('time =', time.time()-t1)
    plt.show()




    Nt_transient = int(10 / dt)
    for T in [25, 1000]:

        model = KS( Nx=Nx,
                    dt=dt,
                    seed=2,
                    initial_amplitude=1.,
                    L=48*np.pi)

        Nt = int(T / model.dt)

        # remove transient
        solution, times = model.time_integrate(Nt=Nt_transient)
        model.update_history(psi=solution[[-1]], t=times[[0]], reset=True)

        comp_time = []
        n_runs = 10
        for _ in range(n_runs):
            t1 = time.time()
            solution, times = model.time_integrate(Nt=Nt)
            t2 = time.time()
            comp_time.append(t2 - t1)

        print(f'Computation time for T={T} over {n_runs} runs: \n ')
        print(f'\t Mean: {np.mean(comp_time):.4f} s')
        print(f'\t Std: {np.std(comp_time):.4f} s')
        print(f'\t Min: {np.min(comp_time):.4f} s')
        print(f'\t Max: {np.max(comp_time):.4f} s')


        model.update_history(psi=solution, t=times)

        KS.visualize_spatiotemporal_hist(model)
        KS.plot_temporal_E(model=model)

    plt.show()


# %%
