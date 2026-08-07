# dynamodels

[![DOI](https://zenodo.org/badge/1326694015.svg)](https://doi.org/10.5281/zenodo.21843588)
[![PyPI](https://img.shields.io/pypi/v/dynamodels)](https://pypi.org/project/dynamodels/)

Dynamical-system models with pre-allocated history tracking and pluggable time
integrators — the modelling core split out of
[romda](https://github.com/andreanovoa/real-time-bias-aware-DA) so it can be
reused on its own. Torch-free; depends only on numpy, scipy, matplotlib,
typeguard and multiprocess.

Tutorial: [`tutorial_dynamodels.ipynb`](tutorial_dynamodels.ipynb) | Interface
documentation: [model protocol](https://andreanovoa.github.io/ntsa/protocol/)

## Install

```bash
pip install dynamodels
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

A model composes a `HistoryTracker` (state/time buffer behind `hist`, `hist_t`,
`update_history()`) with an `Integrator` strategy; subclasses supply
`obs_labels` plus either `time_derivative(t, psi, **params)` (continuous) or
`time_step(Nt)` (discrete map). `m`-member ensembles are supported natively
(`init_ensemble`). Included physical models (`dynamodels.physical`): `Lorenz63`,
`Lorenz96`, `VdP`, `Rijke`, `Annular`, `KS` — the `lorenz63`, `lorenz96` and `rijke`
modules carry measured dominant-Lyapunov tables (module-level `_LAM1_MEASURED`;
`_LAM1_MEASURED_NX10` for Lorenz96) from which instances set `t_lyap`.

## Ecosystem

- [`ntsa`](https://github.com/andreanovoa/ntsa) — nonlinear time-series analysis
  for any dynamodels-style model ([docs](https://andreanovoa.github.io/ntsa/)).
- [romda](https://github.com/andreanovoa/real-time-bias-aware-DA) — bias-aware
  ensemble data assimilation built on top.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
ruff check dynamodels/ tests/
```

Releases: bump `version` in `pyproject.toml`, then `git tag vX.Y.Z && git push --tags`
(publishes to PyPI).

## License

MIT
