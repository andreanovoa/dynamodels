"""Verbosity: dynamodels is silent by default; set_verbose shows its diagnostic messages."""
import logging

import numpy as np
import pytest

import dynamodels
from dynamodels.integrator import _stack_members
from dynamodels.physical import Lorenz63

POOL = 'Initializing multiprocessing pool'


def ensemble_forecast():
    """A forecast of an ensemble of 3 members, which starts the pool of the IVP integrator."""
    model = Lorenz63()
    model.init_ensemble(m=3, std_phi=0.1)
    psi, t = model.time_integrate(Nt=5)
    model.close()
    return psi


@pytest.fixture(autouse=True)
def quiet():
    yield
    dynamodels.set_verbose(False)


def test_silent_by_default(capsys, caplog):
    assert ensemble_forecast().shape == (5, 3, 3)
    out = capsys.readouterr()
    assert out.out == '' and out.err == ''
    assert POOL not in caplog.text


def test_set_verbose_on_and_off(capsys, caplog):
    dynamodels.set_verbose(True)
    dynamodels.set_verbose(True)                                    # idempotent: one handler
    logger = logging.getLogger('dynamodels')
    assert sum(h.get_name() == 'dynamodels.verbose' for h in logger.handlers) == 1
    ensemble_forecast()
    assert POOL in caplog.text
    assert '[dynamodels] ' + POOL in capsys.readouterr().err

    dynamodels.set_verbose(False)
    caplog.clear()
    ensemble_forecast()
    assert POOL not in caplog.text
    assert capsys.readouterr().err == ''
    assert not any(h.get_name() == 'dynamodels.verbose' for h in logger.handlers)


def test_unassigned_keywords_warn():
    with pytest.warns(UserWarning, match='are ignored'):
        Lorenz63(not_a_parameter=1., nor_this_one=2.).close()


def test_failed_member_raises():
    full, short = np.zeros((6, 3)), np.zeros((4, 3))
    assert _stack_members([full, full], 6).shape == (6, 3, 2)
    with pytest.raises(RuntimeError, match=r'member\(s\) \[1\]'):
        _stack_members([full, short], 6)
