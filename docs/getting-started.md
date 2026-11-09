# Getting started

## Install

```bash
pip install dynamodels
```

For development:

```bash
git clone https://github.com/andreanovoa/dynamodels.git
cd dynamodels
pip install -e ".[dev]"
python -m pytest tests/
```

## Quickstart

```python
from dynamodels.physical import Lorenz63

model = Lorenz63(rho=28., dt=0.01)
psi, t = model.time_integrate(Nt=1000)   # (Nt, Nphi, m) states, (Nt,) times
model.update_history(psi, t)
y = model.get_observable_hist()          # (Nt, Nq, m) observables
model.visualize_history()
model.close()                            # release the integrator's pool
```

dynamodels runs silently. For debugging, `dynamodels.set_verbose()` prints its
diagnostic messages on stderr (e.g. the start and the closing of the
multiprocessing pool of an ensemble, which name the model and the integrator),
and `dynamodels.set_verbose(False)` silences them again. Switching off removes
the stderr handler and returns the logger `'dynamodels'` to its default level
(NOTSET), so that the logging configuration of the application applies again.
This also drops a level that the application had set on `'dynamodels'`, with or
without a prior `set_verbose()`, so an application that sets this level sets it
again after `set_verbose(False)`.

Warnings do not depend on the verbosity. When dynamodels ignores a keyword of a
model, or when a call is not recommended, it issues a `UserWarning` that points
at the first line outside dynamodels: the line of the call or, for a call
through a subclass or a wrapper defined outside dynamodels, the line of that
subclass or wrapper. For example, `Lorenz63(sigam=5.)` warns that the misspelt
keyword `sigam` is ignored, and the model keeps the default `sigma`. A keyword
that names a method, e.g. `Lorenz63(copy=1)`, is ignored in the same way, so
that `model.copy()` still works.

A forecast stops with an error when an ensemble member is not finite or
diverges, instead of running on. The error names the model, the member, the
time that the member reached and the reason. A member stops when its solver
takes more than `model.integrator.max_steps` steps (default 10000) within one
output step `dt`. A stiff system that reaches this limit needs a larger
`max_steps`, or an implicit method of `solve_ivp`, e.g.
`model.integrator.method = 'BDF'`. An output step `dt` that is much longer than
the time scale of the system also needs a larger `max_steps`. Both are set on
the integrator of the model, as above. `max_steps` can also be set on the class,
`IVPIntegrator.max_steps = 10**5`, for every model, but the method cannot:
`IVPIntegrator.method = 'BDF'` has no effect, because each integrator holds its
own method.

## The `Model` interface

Every model in `dynamodels.physical` subclasses [`Model`](api/model.md) and supplies:

- `psi0`, `dt`, and a set of named `params` that can be varied or estimated;
- either `time_derivative(t, psi, **params)`, for a continuous system advanced
  by [`IVPIntegrator`](api/integrator.md) (SciPy's `solve_ivp`), or
  `time_step(Nt)`, for a discrete map such as the ETDRK4 scheme used by the
  Kuramoto-Sivashinsky models;
- `obs_labels` and `get_observables`, the sensor model: by convention, the
  leading `Nq` components of the physical state are directly observable.

The state itself lives in a [`HistoryTracker`](api/history.md), pre-allocated
so that repeated calls to `time_integrate` and `update_history` do not
reallocate on every step. `hist` and `hist_t` expose the valid history; `m > 1`
ensembles are supported natively through `init_ensemble`, which estimates the
parameters named in `est_alpha`, from `params`. Any other name raises a
`ValueError` at `init_ensemble`, e.g. a misspelt parameter, the structural
`'nu'` of `KS`, or an attribute that is not a parameter, such as `dt` or the
flame location `'xf'` of `Rijke`. A subclass defined outside dynamodels may
also estimate a class attribute of its own (see `Model.init_ensemble`), but
not an attribute that it sets on the instance only, e.g. `self.gain = 1.` in
its constructor. The subclass registers such an attribute as a parameter by
listing its name in `params`, e.g. `params = [*Lorenz63.params, 'gain']` in
the class body, and sets it before calling `super().__init__()`, which reads
the parameters into `alpha0`. An `init_ensemble` that fails leaves the model
as it was.

`reset_model` restarts a model in place at `t = 0`, from its current state or
from a new `psi0`. The model keeps its ensemble, with the number of members,
the estimated parameters and the observation operator. The model obtains a new
integrator of the same class, with the same `method` and `max_steps`; the old
integrator and its pool close, and other attributes set on the old integrator
are not carried over. Only the values that are overridden change:
`model.reset_model(rho=35.)` changes `rho`, and keeps `dt` and the other
parameters. A structural change, such as a new grid, needs a new model. A new
parameter derives the `filename` again, which replaces a filename set by the
user. A new value of a parameter must be a real number. A reset that fails,
e.g. with `rho=None`, leaves the model as it was, with its integrator and its
pool. A subclass whose constructor derives quantities from `dt` or from a
parameter, e.g. a Lyapunov time, derives them again in `reset_derived`.

Because every model shares this interface, code written against one — a
forecast loop, a plotting routine, a Lyapunov-exponent estimator — runs
unchanged against any other. The sibling package
[`ntsa`](https://andreanovoa.github.io/ntsa/) is built entirely on this
duck-typed protocol; see [Analysing a model](analysis.md).

## Next

- Browse the [models](models/lorenz63.md) for the governing equations and a
  figure of each system's time evolution.
- [Analyse a model](analysis.md) with `ntsa`: delay embeddings, Lyapunov
  exponents, regime classification.
- The [API reference](api/model.md) documents `Model`, the three integrator
  strategies, and `dynamodels.utils`.
- The repository's
  [`tutorial_dynamodels.ipynb`](https://github.com/andreanovoa/dynamodels/blob/main/tutorial_dynamodels.ipynb)
  walks through every model interactively.
