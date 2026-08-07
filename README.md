# dynamodels

Dynamical-system models with pre-allocated history tracking and pluggable time
integrators. This is the modelling core split out of
[romda](https://github.com/andreanovoa/real-time-bias-aware-DA) (real-time
bias-aware data assimilation) so it can be reused on its own — it is torch-free
and depends only on numpy, scipy, matplotlib, typeguard and multiprocess.

```bash
pip install dynamodels            # once released; until then:
pip install "dynamodels @ git+https://github.com/andreanovoa/dynamodels"
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

## Architecture

`Model` composes two pieces:

- **`HistoryTracker`** — pre-allocated, grow-on-demand state/time buffer.
  Access via `hist`, `hist_t`, `current_state`, `current_time`,
  `update_history()`.
- **`Integrator`** strategy — `IVPIntegrator` (scipy `solve_ivp`, with a
  multiprocessing pool for ensembles), `DiscreteIntegrator` (calls
  `model.time_step()`; maps such as KS), `ConstantIntegrator` (frozen state).

A model subclass supplies `obs_labels` plus either
`time_derivative(t, psi, **params)` (continuous → IVP) or `time_step(Nt)`
(discrete map). Sweepable parameters are declared in the `params` class
attribute and set through the constructor; change parameters by
re-instantiation.

### Included physical models (`dynamodels.physical`)

| Class | System |
| --- | --- |
| `Lorenz63` | Lorenz 1963 convection ODEs |
| `Lorenz96` | Lorenz 1996 ring lattice (any `Nx`) |
| `VdP` | Van der Pol oscillator (thermoacoustic limit-cycle surrogate) |
| `Rijke` | Rijke tube: Galerkin acoustics + Chebyshev advection flame model |
| `Annular` | annular combustor azimuthal-mode model |
| `KS` | Kuramoto–Sivashinsky (discrete spectral map) |

`Lorenz63`, `Lorenz96` and `Rijke` carry measured dominant-Lyapunov tables
(`_LAM1_MEASURED`): inside the chaotic range of the sweep parameter, instances
get `t_lyap = 1/λ1` log-interpolated at the constructed value (class attributes
keep the historical constants).

### Ensembles

`m > 1` ensemble members are supported throughout (`init_ensemble`,
`mean_vector_to_ensemble`); the IVP integrator forecasts members through a
multiprocessing pool. Everything downstream of `hist` carries the trailing
member axis `(Nt, N, m)`.

## Ecosystem

- [`ntsa`](https://github.com/andreanovoa/ntsa) — nonlinear time-series
  analysis (Lyapunov spectra, delay embeddings, regime classification) for any
  `dynamodels`-style model.
- [romda](https://github.com/andreanovoa/real-time-bias-aware-DA) — bias-aware
  ensemble data assimilation built on top (estimators, bias estimators,
  data-driven ESN/POD models).

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
ruff check dynamodels/ tests/
```

## Releasing (maintainer note)

Releases publish to PyPI via GitHub Actions trusted publishing on version tags:
configure a trusted publisher for `andreanovoa/dynamodels` (workflow
`release.yml`, environment `pypi`) at pypi.org, then
`git tag v0.1.0 && git push --tags`.

## License

MIT
