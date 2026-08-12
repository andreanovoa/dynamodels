"""dynamodels — dynamical-system models with history tracking and pluggable integrators.

The `Model` base class composes a pre-allocated state/time history
(`HistoryTracker`) with an `Integrator` strategy (`IVPIntegrator` for
continuous ODEs via scipy `solve_ivp`, `DiscreteIntegrator` for maps,
`ConstantIntegrator` for frozen states). A model subclass supplies
`obs_labels` plus either `time_derivative(t, psi, **params)` or
`time_step(Nt)`. Ready-made physical models live in `dynamodels.physical`:
Lorenz63, Lorenz96, Van der Pol, Rijke, Annular, KS.

Originally the `romda.models` core (real-time bias-aware data assimilation);
split out so the modelling layer is reusable on its own.
"""

from . import physical
from .history import HistoryTracker
from .integrator import ConstantIntegrator, DiscreteIntegrator, Integrator, IVPIntegrator
from .model import Model

__version__ = "0.3.1"

__all__ = [
    "Model",
    "HistoryTracker",
    "Integrator",
    "IVPIntegrator",
    "DiscreteIntegrator",
    "ConstantIntegrator",
    "physical",
]
