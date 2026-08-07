"""Shared helpers vendored from romda.utils (pure numpy/scipy, no romda dependency)."""

import inspect
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


def allowed_kwargs_for_func(func, kwargs):
    """Return the subset of `kwargs` that are valid parameter names of `func`."""
    sig = inspect.signature(func)
    accepted = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}




def normalized_time(
    reference_t: float, *times
) -> tuple[Sequence[np.ndarray | None], str]:
    """Rescale each array in `times` by `reference_t` and build a matching axis label."""
    if reference_t == 1.0:
        return times, '$t$'
    return [t / reference_t if t is not None else t for t in times], f'$t/{reference_t}$'


def normalized_y(
    reference_y: float | np.ndarray,
    y_labels,
    *ys,
) -> tuple[Sequence[np.ndarray | None], Sequence[str | None]]:
    """Divide each array in `ys` by `reference_y` (scalar or per-observable) and
    annotate `y_labels` accordingly."""
    Nys = [y.shape[1] for y in ys if y is not None]
    assert Nys, 'At least one y must be provided.'
    Ny = Nys[0]
    if isinstance(reference_y, (int, float)):
        if reference_y == 1.0:
            return ys, y_labels
        reference_y = reference_y * np.ones(Ny)
    reference_y = np.asarray(reference_y)
    ref = reference_y[np.newaxis, :, np.newaxis]
    ys_norm = [y.copy() / ref if y is not None else y for y in ys]
    lbls = [f'{y_labels[qi]} / ${reference_y[qi]}$' for qi in range(Ny)]
    return ys_norm, lbls


def normalized_alpha(alpha, alpha_keys, alpha_labels, reference_a=None):
    """Divide each estimated parameter in `alpha` by its `reference_a` value (default
    1.0) and annotate `alpha_labels` accordingly."""
    reference_alpha = {key: 1.0 for key in alpha_keys}
    # default to the raw key for estimated parameters without a pretty label (e.g. ESN 'svd_0')
    alpha_lbls = {**{key: key for key in alpha_keys}, **alpha_labels}
    alpha = alpha.copy()
    if reference_a is not None and isinstance(reference_a, dict):
        for ai, key in enumerate(alpha_keys):
            ref = reference_a.get(key, 1.0)
            reference_alpha[key] = ref
            if ref != 1.0:
                alpha_lbls[key] += f' / {ref}'
            alpha[:, ai] /= ref
    return alpha, alpha_lbls



def mean_vector_to_ensemble(rng: np.random.Generator,
                            mean_vec: NDArray[np.floating],
                            std: float | NDArray[np.floating] | list[float] | dict[str, float | list[float] | tuple[float, float]],
                            m: int,
                            method: str = 'uniform',
                            ensure_mean_at_init: bool = False) -> np.ndarray:
    """Perturb a mean state/parameter vector to build an initial ensemble.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator used to draw the perturbations.
    mean_vec : ndarray
        Mean vector to perturb, shape ``(state_dim,)``.
    std : float, ndarray, list or dict
        Uncertainty around `mean_vec`. A float or array gives a relative standard
        deviation applied multiplicatively to every component of `mean_vec`. A
        dict (one entry per estimated parameter) gives, for ``method='uniform'``,
        ``[min, max]`` bounds per parameter, or, for ``method='normal'``, a
        location/scale derived from those bounds (or a single value).
    m : int
        Ensemble size.
    method : {'uniform', 'normal'}, optional
        Sampling distribution. Default ``'uniform'``.
    ensure_mean_at_init : bool, optional
        If True, overwrite the first ensemble member with the unperturbed mean.
        Default False.

    Returns
    -------
    np.ndarray
        Ensemble array, shape ``(state_dim, m)``.
    """
    if method not in ['uniform', 'normal']:
        raise ValueError(f'Distribution "{method}" not supported. Choose "uniform" or "normal".')

    mean_vec = np.asarray(mean_vec.copy())
    if mean_vec.ndim > 1:
        mean_vec = np.atleast_1d(mean_vec.squeeze())  # Ensure mean_vec is 1D


    # Case 1: std is a dictionary (for estimated parameters 'alpha')
    if isinstance(std, dict):
        ensemble_ = []
        for sa in std.values():
            if method == 'uniform':
                # For uniform, std values are [min_val, max_val]
                ensemble_.append(rng.uniform(low=sa[0], high=sa[1], size=m)) #type: ignore
            else: # normal
                # Use mean of bounds as location, and half the range as a heuristic scale (std)
                loc = np.mean(sa)
                if isinstance(sa, list) and len(sa) == 2:
                    scale = (sa[1] - sa[0]) / 4.0
                else:
                    scale = loc * 0.5
                ensemble_.append(rng.normal(loc=loc, scale=scale, size=m))
        ensemble_ = np.array(ensemble_) # Shape: (num_params, m)

    # Case 2: std is a single float or a different std for each component (relative standard deviation for state or parameters)
    elif isinstance(std, float) or isinstance(std, np.ndarray):
        if method == 'uniform':
            # ensure std is an array with. compatible shape
            if isinstance(std, float):
                std = std * np.ones_like(mean_vec)
            if std.ndim == 1:
                std = std[:, np.newaxis]


            perturbation = 1.0 + rng.uniform(-std, std, size=(mean_vec.size, m))
            ensemble_ = mean_vec[:, np.newaxis] * perturbation

        else: # normal (using multivariate normal for state vector)

            # Multiplicative uniform perturbation: mean * (1 +/- std)
            # print(f'Creating normal ensemble with std={std} for mean_vec of shape {mean_vec.shape} and m={m}')

            if np.iscomplexobj(mean_vec):
                # Handle complex state by perturbing real and imaginary parts independently
                # ensure we have a numpy array so static type checkers accept real/imag access
                real_mu = np.real(mean_vec)
                imag_mu = np.imag(mean_vec)
                real_part = rng.multivariate_normal(real_mu, np.diag((real_mu * std) ** 2), size=m).T
                imag_part = rng.multivariate_normal(imag_mu,  np.diag((imag_mu * std) ** 2), size=m).T
                ensemble_ = real_part + 1j * imag_part
            else:
                # Covariance matrix is diagonal, perturbation scaled by mean and relative std
                cov = np.diag((mean_vec * std) ** 2)
                ensemble_ = rng.multivariate_normal(mean_vec, cov, size=m).T

    else:
        raise TypeError(f'Initial std must be a float or a dict, not {type(std)}')


    # Replace the first member with the unperturbed mean
    if ensure_mean_at_init and ensemble_ is not None:
        ensemble_[:, 0] = mean_vec

    return ensemble_


def Cheb(Nc, lims=(0, 1), getg=False):
    """Compute the Chebyshev collocation derivative matrix and grid.

    Parameters
    ----------
    Nc : int
        Number of Chebyshev intervals; the grid has ``Nc + 1`` points.
    lims : tuple of float, optional
        Domain limits. If ``lims[0] == 0``, the grid is mapped from $[-1, 1]$ to
        $[0, 1]$. Default ``(0, 1)``.
    getg : bool, optional
        If True, also return the grid points. Default False.

    Returns
    -------
    D : np.ndarray
        Chebyshev differentiation matrix, shape ``(Nc + 1, Nc + 1)``.
    g : np.ndarray, optional
        Chebyshev grid points, shape ``(Nc + 1,)``. Only returned if `getg` is True.
    """
    g = - np.cos(np.pi * np.arange(Nc + 1, dtype=float) / Nc)
    c = np.hstack([2., np.ones(Nc - 1), 2.]) * (-1) ** np.arange(Nc + 1)
    X = np.outer(g, np.ones(Nc + 1))
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(Nc + 1))
    D -= np.diag(D.sum(1))

    # Modify
    if lims[0] == 0:
        g = (g + 1.) / 2.
    if getg:
        return D, g
    else:
        return D


def interpolate(t_y, y, t_eval, fill_values: tuple[float, float] | str | None = None):
    """Linearly interpolate `y(t_y)` (along its leading axis) at `t_eval`.

    Parameters
    ----------
    t_y : array-like
        Time points of `y`, shape ``(T,)``.
    y : array-like
        Values to interpolate, shape ``(T, ...)``.
    t_eval : array-like
        Time points at which to evaluate the interpolant.
    fill_values : tuple of float or str, optional
        Value(s) used outside the range of `t_y`. Defaults to ``(y[0], y[-1])``
        (constant extrapolation); see `scipy.interpolate.interp1d`'s
        ``fill_value`` for accepted values.

    Returns
    -------
    np.ndarray
        Interpolated values at `t_eval`, shape ``(len(t_eval), ...)``.
    """
    if fill_values is None:
        fill_values = (y[0], y[-1])

    interpolator = interp1d(t_y, y,
                            axis=0,  # interpolate along columns
                            bounds_error=False,
                            kind='linear',
                            fill_value=fill_values # type: ignore
                            )
    return interpolator(t_eval)
