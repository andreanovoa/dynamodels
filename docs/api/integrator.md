# `Integrator`

Three time-integration strategies, selected at construction and shared by
every model: `IVPIntegrator` for continuous systems (SciPy's `solve_ivp`,
with a lazily started pool of `min(m, cpu_count)` worker processes for
`m > 1` ensembles),
`DiscreteIntegrator` for fixed-step maps such as the ETDRK4 scheme, and
`ConstantIntegrator` for a frozen state.

The method of `solve_ivp` is set on the integrator of a model, e.g.
`model.integrator.method = 'DOP853'` (default `'RK45'`), and applies to a
single member and to an ensemble. The method is an attribute of each
integrator, set by its constructor, so `IVPIntegrator.method = 'DOP853'` on
the class has no effect. In contrast, `max_steps` (below) is a class
attribute: `model.integrator.max_steps = 10**5` changes the limit of one
model, and `IVPIntegrator.max_steps = 10**5` changes the default of every
model.

An `IVPIntegrator` forecast stops with an error, instead of stalling, when a
member is not finite or diverges. A member whose initial state holds a NaN or
an inf raises a `ValueError` before any member is integrated. A member that
stops before the end of the forecast raises a `RuntimeError`, which names the
model, the member, the last output time that the member reached and the
reason. A diverged state can shrink the time scale of the system: Lorenz63 at
x = 1e200 needs solver steps of about 1e-201, i.e. about 1e200 steps per
output step. `IVPIntegrator.max_steps` (default 10000 solver steps per output
step) stops such a member in less than a second.

::: dynamodels.integrator.Integrator

::: dynamodels.integrator.IVPIntegrator

::: dynamodels.integrator.DiscreteIntegrator

::: dynamodels.integrator.ConstantIntegrator
