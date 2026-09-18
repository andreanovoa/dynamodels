# Lorenz63

The Lorenz (1963) system is the standard low-order benchmark for deterministic
chaos: three coupled ordinary differential equations,

$$
\dot{x} = \sigma (y - x), \qquad
\dot{y} = x (\rho - z) - y, \qquad
\dot{z} = x y - \beta z,
$$

which model convective roll motion in a truncated Rayleigh-Benard problem. At
the classical parameters ($\sigma=10$, $\rho=28$, $\beta=8/3$), the system is
chaotic with leading Lyapunov exponent $\lambda_1 \approx 0.906$, so two
trajectories starting a distance $\epsilon$ apart diverge to order-one
separation within a Lyapunov time $1/\lambda_1 \approx 1.1$.

![Lorenz63 observable time evolution](../img/lorenz63.png)

*Time evolution of $x$, $y$ and $z$ for $\rho=28$, past the initial transient.
Left: the full run. Right: the last four Lyapunov times, showing the
characteristic double-lobe switching of the attractor.*

## Quickstart

```python
from dynamodels.physical import Lorenz63

model = Lorenz63(rho=28., sigma=10., beta=8./3, dt=0.02)
psi, t = model.time_integrate(Nt=5000)
model.update_history(psi, t)
model.visualize_attractor()   # the classical butterfly, in 3-D and its three projections
model.close()
```

`observe_dims` (default `[0, 1, 2]`, all three states) selects which
components are observable; `Lorenz63(observe_dims=[0])` restricts the model to
observing $x$ alone, as in a partial-observation data-assimilation setup.

## Nonlinear diagnostics

![ntsa characterization of Lorenz63](../img/ntsa_characterize_lorenz63.png)

*Diagnostics from [`ntsa.characterize`](../analysis.md), left to right: the
observable time series with a zoomed inset; power spectral density; the 3-D
delay-embedded portrait; the first-return map of the maxima; a
plane-crossing Poincare section; a recurrence plot; a 3-D classical-MDS
embedding of the full state; and the Lyapunov spectrum, confirming the
chaotic classification.*

## Reference

Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the
Atmospheric Sciences*, 20(2), 130-141.

## API

::: dynamodels.physical.lorenz63.Lorenz63
