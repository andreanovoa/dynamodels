"""Smoke tests for the physical models + the package's torch-free guarantee."""
import sys

import numpy as np

import dynamodels
from dynamodels.physical import KS, Annular, Lorenz63, Lorenz96, Rijke, VdP


def test_torch_free_import():
    # the whole point of the split: importing the modelling core must not drag
    # in any ML stack
    assert 'torch' not in sys.modules, 'importing dynamodels pulled in torch'
    assert 'skopt' not in sys.modules, 'importing dynamodels pulled in scikit-optimize'
    assert dynamodels.__version__


def test_models_integrate_smoke():
    for cls, kwargs in [(Lorenz63, {}), (Lorenz96, dict(Nx=10)), (VdP, {}),
                        (Rijke, {}), (Annular, {}), (KS, {})]:
        model = cls(**kwargs)
        psi, t = model.time_integrate(Nt=10)
        assert psi.shape[0] == len(t) and np.all(np.isfinite(psi)), f'{cls.__name__} integration failed'
        model.update_history(psi, t)
        y = model.get_observable_hist()
        assert y.shape[1] == model.Nq, f'{cls.__name__} observables wrong shape'
        assert len(model.obs_labels) == model.Nq
        model.close()


def test_t_lyap_tables():
    # measured-lambda1 tables in the physical model files: instances get
    # t_lyap = 1/lambda1(sweep param) inside the chaotic range, class constants
    # and off-table configurations are untouched
    assert np.isclose(Lorenz63.t_lyap, 0.9056 ** -1), 'class access must stay the constant'
    ms = [Lorenz63(), Lorenz63(rho=10.),
          Lorenz96(Nx=10, F=8.), Lorenz96(Nx=10, F=4.8), Lorenz96(Nx=10, F=2.),
          Lorenz96(Nx=40, F=8.), Rijke(beta=12.), Rijke(beta=4.)]
    expect = [1 / 0.917, 0.9056 ** -1,
              1 / 1.184, None, 1.67 ** -1,
              1.67 ** -1, 1 / 161.0, 0.02]
    for m, e in zip(ms, expect):
        if e is None:  # F=4.8: log-interpolated between the F=4.6 and F=5 entries
            assert 1 / 0.058 < m.t_lyap < 1 / 0.0391, f'{m.name} t_lyap = {m.t_lyap}'
        else:
            assert np.isclose(m.t_lyap, e), f'{type(m).__name__} t_lyap = {m.t_lyap}, expected {e}'
        m.close()
