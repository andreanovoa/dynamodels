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
ensembles are supported natively through `init_ensemble`.

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
