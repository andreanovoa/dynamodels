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

The package is silent by default; `set_verbose()` shows its diagnostic
messages (the logger 'dynamodels') for debugging.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import physical  # noqa: E402
from .history import HistoryTracker  # noqa: E402
from .integrator import ConstantIntegrator, DiscreteIntegrator, Integrator, IVPIntegrator  # noqa: E402
from .model import Model  # noqa: E402
from .verbose import set_verbose  # noqa: E402

__version__ = "0.6.0"

__all__ = [
    "Model",
    "HistoryTracker",
    "Integrator",
    "IVPIntegrator",
    "DiscreteIntegrator",
    "ConstantIntegrator",
    "physical",
    "set_verbose",
]
