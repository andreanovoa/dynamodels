# `Integrator`

Three time-integration strategies, selected at construction and shared by
every model: `IVPIntegrator` for continuous systems (SciPy's `solve_ivp`,
with a lazily created multiprocessing pool for `m > 1` ensembles),
`DiscreteIntegrator` for fixed-step maps such as the ETDRK4 scheme, and
`ConstantIntegrator` for a frozen state.

::: dynamodels.integrator.Integrator

::: dynamodels.integrator.IVPIntegrator

::: dynamodels.integrator.DiscreteIntegrator

::: dynamodels.integrator.ConstantIntegrator
