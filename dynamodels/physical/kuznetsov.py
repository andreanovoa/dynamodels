import numpy as np

from ..integrator import IVPIntegrator
from ..model import Model


# %% =================================== KUZNETSOV OSCILLATOR ============================================== %% #
class Kuznetsov(Model):
    r"""Kuznetsov oscillator — autonomous generator of quasiperiodic oscillations.

    Three-dimensional system displaying periodic, quasiperiodic and chaotic
    behaviour as a function of the parameters $[\lambda, \omega_0, \mu]$:

    $$
    \dot{x} = y, \qquad
    \dot{y} = y \left( \lambda + z + x^2 - \tfrac{1}{2} x^4 \right) - \omega_0^2 x, \qquad
    \dot{z} = \mu - x^2.
    $$

    With the default parameters ($\lambda = 0$, $\omega_0 = 2\pi$, $\mu = 1$)
    the attractor is a two-frequency quasiperiodic torus.

    References
    ----------
    Kuznetsov, Kuznetsov & Stankevich (2010). A simple autonomous quasiperiodic
    self-oscillator. *Commun. Nonlinear Sci. Numer. Simul.*, 15, 1676–1681.
    [DOI: 10.1016/j.cnsns.2009.06.027](https://doi.org/10.1016/j.cnsns.2009.06.027).
    """

    t_transient = 100.   # fast period is 2 pi / omega0 = 1; slow beats need many of them
    t_CR = 4.
    t_lyap = t_CR        # no measured-lambda1 table yet (default case is quasiperiodic)

    Nq = 3

    lam = 0.             # linear growth rate 
    omega0 = 2 * np.pi   # natural frequency [rad/s]
    mu = 1.              # slow-variable drive

    params = ['lam', 'omega0', 'mu']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):

        psi0 = model_dict.pop('psi0', np.array([0.1, 0., 0.]))
        dt = model_dict.pop('dt', 0.01)

        super().__init__(psi0=psi0, dt=dt, integrator_class=IVPIntegrator, **model_dict)

        self.alpha_labels = dict(lam='$\\lambda$', omega0='$\\omega_0$', mu='$\\mu$')

    @property
    def state_labels(self):
        return ['$x$', '$y$', '$z$']

    # _______________ Kuznetsov specific properties and methods ________________ #
    @property
    def obs_labels(self):
        return self.state_labels[:self.Nq]

    @staticmethod
    def time_derivative(t, psi, lam, omega0, mu):
        x, y, z = psi[:3]
        dx = y
        dy = y * (lam + z + x ** 2 - 0.5 * x ** 4) - omega0 ** 2 * x
        dz = mu - x ** 2
        return (dx, dy, dz) + (0,) * (len(psi) - 3)
