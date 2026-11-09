"""Verbosity: dynamodels is silent by default; set_verbose shows its diagnostic messages."""
import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest

import dynamodels
import dynamodels.integrator
from dynamodels.integrator import IVPIntegrator
from dynamodels.physical import Lorenz63

POOL = 'Lorenz63: starting a pool of'
CLOSING = '{}: closing the pool of worker processes ({}).'


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


def application_defaults(caplog):
    """Pin the logging of an application that does not configure it: the root logger at WARNING, whatever the log
    level of the pytest run (e.g. ``pytest -o log_level=DEBUG``). caplog captures every record that reaches the root
    logger, and restores both levels after the test."""
    caplog.set_level(logging.WARNING)
    caplog.handler.setLevel(logging.NOTSET)


def test_silent_by_default(capsys, caplog):
    application_defaults(caplog)
    assert ensemble_forecast().shape == (5, 3, 3)
    out = capsys.readouterr()
    assert out.out == '' and out.err == ''
    assert POOL not in caplog.text


def test_set_verbose_on_and_off(capsys, caplog):
    application_defaults(caplog)
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
    assert logger.level == logging.NOTSET                           # the default, not WARNING
    assert logger.propagate and not logger.disabled                 # the records reach the application's handlers


@pytest.mark.parametrize('calls', [(True, False), (False,)], ids=['on-off', 'off-without-on'])
def test_set_verbose_off_returns_the_logger_to_the_application(capsys, caplog, calls):
    """Off returns 'dynamodels' to the default level (NOTSET), with or without a prior on: a level that the
    application had set on 'dynamodels' is dropped, and the configuration of the root logger applies."""
    logger = logging.getLogger('dynamodels')
    logger.setLevel(logging.ERROR)                                  # a level that the application had set
    for on in calls:
        dynamodels.set_verbose(on)
    assert logger.level == logging.NOTSET and logger.propagate and not logger.disabled
    assert logger.getEffectiveLevel() == logging.getLogger().getEffectiveLevel()

    caplog.set_level(logging.DEBUG)                                 # the application shows debug messages
    ensemble_forecast()
    assert any(r.name.startswith('dynamodels') and r.levelno < logging.WARNING for r in caplog.records)
    assert capsys.readouterr().err == ''                            # through its handlers, not set_verbose's


@pytest.mark.parametrize('calls', [(False,), (False, False), (True, False, False)],
                         ids=['off-without-on', 'off-twice', 'on-off-off'])
def test_set_verbose_off_is_safe_without_on_and_when_repeated(capsys, caplog, calls):
    """Off without a prior on, or repeated, raises no error: 'dynamodels' stays at the default level (NOTSET), the
    NullHandler that the package adds at import is kept, and the package stays silent."""
    application_defaults(caplog)
    logger = logging.getLogger('dynamodels')
    handlers = list(logger.handlers)
    assert logger.level == logging.NOTSET
    assert any(type(h) is logging.NullHandler for h in handlers)
    assert not any(h.get_name() == 'dynamodels.verbose' for h in handlers)

    for on in calls:
        dynamodels.set_verbose(on)
    assert logger.level == logging.NOTSET and logger.propagate and not logger.disabled
    assert logger.handlers == handlers

    ensemble_forecast()
    assert POOL not in caplog.text
    out = capsys.readouterr()
    assert out.out == '' and out.err == ''


def test_unassigned_keywords_warn():
    with pytest.warns(UserWarning, match='are ignored'):
        Lorenz63(not_a_parameter=1., nor_this_one=2.).close()


def test_pool_start_names_the_model_workers_and_members(caplog):
    with caplog.at_level(logging.DEBUG, logger='dynamodels'):
        ensemble_forecast()
    # min(3, cpu_count) workers: one pool of worker processes, not several pools
    assert re.search(r'Lorenz63: starting a pool of (1 worker process|[23] worker processes) for the 3 ensemble '
                     r'members \(IVPIntegrator\)\.', caplog.text)
    assert 'pools' not in caplog.text


def test_each_pool_closing_is_logged(caplog):
    # a stand-in pool, so that no process starts: close, copy, create_long_timeseries and reset_model log one closing
    model = Lorenz63()
    fake = SimpleNamespace(terminate=lambda: None, join=lambda: None)
    with caplog.at_level(logging.DEBUG, logger='dynamodels'):
        model.integrator._pool = fake
        model.close()
        model.integrator._pool = fake
        twin = model.copy()
        model.integrator._pool = fake
        model.create_long_timeseries(Nt=2)
        old = model.integrator
        old._pool = fake
        model.reset_model()                                         # replaces the integrator, and closes the old one
        model.copy().close()                                        # no pool: nothing to log
    closings = [r.getMessage() for r in caplog.records if 'closing the pool' in r.getMessage()]
    assert closings == [CLOSING.format('Lorenz63', 'IVPIntegrator')] * 4
    assert old._pool is None and twin.integrator._pool is None


def pool_messages(caplog):
    """The start and the closing messages of the pools, in the order logged."""
    return [r.getMessage() for r in caplog.records if ' pool of ' in r.getMessage()]


def starting(name, m, integrator='IVPIntegrator'):
    """The start message of the pool of `min(m, cpu_count)` workers of an ensemble of `m` members."""
    n = min(m, dynamodels.integrator.mp.cpu_count())
    return (f'{name}: starting a pool of {n} worker process{"" if n == 1 else "es"} for the {m} ensemble members '
            f'({integrator}).')


class TwinIVP(IVPIntegrator):
    """A subclass of IVPIntegrator, which the pool messages name."""


def test_pool_messages_name_a_user_set_model_name(caplog):
    model = Lorenz63()
    model.name = 'L63 twin'
    model.init_ensemble(m=3, std_phi=0.1)
    with caplog.at_level(logging.DEBUG, logger='dynamodels'):
        model.time_integrate(Nt=2)
        model.close()
    assert pool_messages(caplog) == [starting('L63 twin', 3), CLOSING.format('L63 twin', 'IVPIntegrator')]


def test_pool_messages_name_the_integrator_and_one_worker_process(caplog, monkeypatch):
    """With one CPU, the pool has one worker process; the messages name the class of the integrator, and the closing
    does not announce a new pool, which an averaged forecast (averaged=True) does not start."""
    monkeypatch.setattr(dynamodels.integrator.mp, 'cpu_count', lambda: 1)
    model = Lorenz63()
    model.integrator = TwinIVP(model)
    model.init_ensemble(m=3, std_phi=0.1)
    with caplog.at_level(logging.DEBUG, logger='dynamodels'):
        model.time_integrate(Nt=2)
        model.close()
        model.time_integrate(Nt=2, averaged=True)                   # no pool
        model.close()
    assert pool_messages(caplog) == ['Lorenz63: starting a pool of 1 worker process for the 3 ensemble members '
                                     '(TwinIVP).', 'Lorenz63: closing the pool of worker processes (TwinIVP).']


def test_a_new_number_of_members_closes_the_pool(caplog):
    """A pool serves one number of members. init_ensemble with a new m closes it, and the next forecast starts a pool
    for the new members; the same m keeps it. A history reset with a new number of members closes the pool at the
    next forecast."""
    model = Lorenz63()
    with caplog.at_level(logging.DEBUG, logger='dynamodels'):
        model.init_ensemble(m=2, std_phi=0.1)
        model.time_integrate(Nt=2)
        pool = model.integrator._pool
        model.init_ensemble(m=2, std_phi=0.1)                       # its transient forecast uses the pool
        assert model.integrator._pool is pool
        model.init_ensemble(m=6, std_phi=0.1)
        assert model.integrator._pool is None
        model.time_integrate(Nt=2)
        model.update_history(model.current_state[None, :, :3], t=np.array([model.current_time]), reset=True)
        assert model.m == 3
        model.time_integrate(Nt=2)
        model.close()
    closing = CLOSING.format('Lorenz63', 'IVPIntegrator')
    assert pool_messages(caplog) == [starting('Lorenz63', 2), closing, starting('Lorenz63', 6), closing,
                                     starting('Lorenz63', 3), closing]
