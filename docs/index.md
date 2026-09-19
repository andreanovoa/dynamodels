# dynamodels

[![PyPI](https://img.shields.io/pypi/v/dynamodels.svg)](https://pypi.org/project/dynamodels/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21843588-blue.svg)](https://doi.org/10.5281/zenodo.21843588)

A `Model` couples a governing equation to a pre-allocated state history and a pluggable time integrator. Every physical model in the package — three low-order oscillators, two spatially extended PDEs, and the Lorenz systems — implements the same interface: 
- `time_integrate(Nt)` advances the state,
- `get_observable_hist()` returns what a sensor would measure, and
- `visualize_history()` plots both. 
The package itself depends only on numpy, scipy, matplotlib and typeguard.

![Lorenz63 attractor](img/Lorenz_ergodic.gif)


## Where to go

- [Getting started](getting-started.md) — install, the five-line quickstart, the `Model` interface.
- [Example usage](tutorials/tutorial_dynamodels.html) — the `Model` interface end to end in an
  executed notebook: integration, the history buffer, observables, ensembles and integrators.
- Available Models — one page per physical model, each with the governing equations and a figure
  of its time evolution: [Lorenz63](models/lorenz63.md), [Lorenz96](models/lorenz96.md),
  [Van der Pol](models/van_der_pol.md), [Rijke tube](models/rijke.md),
  [Annular combustor](models/annular.md), [Kuramoto-Sivashinsky (1-D)](models/ks.md),
  [Kuramoto-Sivashinsky (2-D)](models/ks2d.md), [Kuznetsov oscillator](models/kuznetsov.md).
- API reference — [`Model`](api/model.md), [`Integrator`](api/integrator.md),
  [`HistoryTracker`](api/history.md), [`utils`](api/utils.md).
- Learn more — [Tutorials on thermoacoustics](tutorials.md), the Rijke tube and annular
  combustor notebooks; [Characterizing a model](analysis.md), nonlinear time-series diagnostics
  with [`ntsa`](https://andreanovoa.github.io/ntsa/), the sibling package built on this one.

## Ecosystem

- [`ntsa`](https://andreanovoa.github.io/ntsa/) — nonlinear time-series analysis
  (Lyapunov exponents, delay embeddings, regime classification, bifurcation sweeps)
  for any model that follows the `dynamodels` protocol.
- [romda](https://andreanovoa.github.io/real-time-bias-aware-DA/) — bias-aware
  ensemble data assimilation built on top of both.

## Citing

If a model in this package is central to your work, please cite its original
reference, listed on that model's page. To cite the package itself:

```bibtex
@software{novoa2026dynamodels,
  author    = {N{\'o}voa, Andrea},
  title     = {dynamodels -- dynamical systems models},
  year      = {2026},
  version   = {v0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21843589},
  url       = {https://doi.org/10.5281/zenodo.21843589},
}
```
