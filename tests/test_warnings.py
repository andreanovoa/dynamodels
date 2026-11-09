"""User-facing warnings of Model: an ignored keyword always warns, and each warning points at the first line outside
dynamodels. That line is the line of the call or, for a call through a subclass or a wrapper defined outside dynamodels,
the line of that subclass or wrapper."""
import functools
import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import dynamodels.model
import dynamodels.utils
from dynamodels.physical import KS, KS2D, Annular, Kuznetsov, Lorenz63, Lorenz96, Rijke, VdP


def _next_line():
    """Number of the line after the line of the caller."""
    return inspect.currentframe().f_back.f_lineno + 1


def _ignored(records):
    """The warnings about ignored keywords among `records`."""
    return [w for w in records if 'are ignored' in str(w.message)]


@pytest.fixture(params=[True, False], ids=['skip_file_prefixes', 'frame_walk'])
def attribution(request, monkeypatch):
    """Both attribution paths: skip_file_prefixes (Python 3.12+) and the frame walk (Python 3.10 and 3.11)."""
    if request.param and not dynamodels.utils._HAS_SKIP_FILE_PREFIXES:
        pytest.skip('warnings.warn(skip_file_prefixes=...) needs Python 3.12+')
    monkeypatch.setattr(dynamodels.utils, '_HAS_SKIP_FILE_PREFIXES', request.param)


@pytest.mark.parametrize('cls, kwargs, typo', [(Lorenz63, dict(sigam=5.), 'sigam'),
                                               (Lorenz96, dict(Nx=10, Fx=8.), 'Fx'),
                                               (VdP, dict(betta=50.), 'betta'),
                                               (Kuznetsov, dict(omega=1.), 'omega'),
                                               (Rijke, dict(betaa=1.), 'betaa'),
                                               (Annular, dict(kapa=1e-4), 'kapa'),
                                               (KS, dict(Nx=32, nuu=1.), 'nuu'),
                                               (KS2D, dict(Nx=16, Ny=16, nu_1=0.5), 'nu_1')],
                         ids=['Lorenz63', 'Lorenz96', 'VdP', 'Kuznetsov', 'Rijke', 'Annular', 'KS', 'KS2D'])
def test_one_ignored_keyword_warns_at_the_caller(attribution, cls, kwargs, typo):
    match = re.escape(f"{cls.__name__}: keyword(s) ['{typo}'] are not attributes")
    with pytest.warns(UserWarning, match=match) as rec:
        line = _next_line()
        model = cls(**kwargs)
    model.close()
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, line)


class _Wrapper(Lorenz63):
    """A subclass defined outside dynamodels that passes its keywords on, as the estimators of dependents do."""

    def __init__(self, **kwargs):
        self.line = _next_line()
        super().__init__(**kwargs)


def test_through_a_subclass_the_warning_points_at_the_subclass(attribution):
    """The first line outside dynamodels is the super().__init__ of the subclass, not the line of the call."""
    with pytest.warns(UserWarning, match=re.escape("_Wrapper: keyword(s) ['sigam'] are not attributes")) as rec:
        model = _Wrapper(sigam=5.)
    model.close()
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, model.line)


def _shift(self, value, by):
    return value + by


class _Custom(Lorenz63):
    """A user subclass with a property that has a setter, which a keyword sets, and with methods, which it does not
    set: a classmethod, and a singledispatchmethod and a partialmethod, which are not callable themselves."""

    _gain = 1.

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = value

    @classmethod
    def build(cls):
        return cls()

    @functools.singledispatchmethod
    def scaled(self, value):
        return 2 * value

    shifted = functools.partialmethod(_shift, by=1)


@pytest.mark.parametrize('cls, kwargs, method',
                         [(Lorenz63, {}, 'copy'), (Lorenz63, {}, 'reset_model'), (Lorenz63, {}, 'time_derivative'),
                          (Lorenz63, {}, 't_lyap_from_table'), (_Custom, {}, 'build'),
                          (_Custom, {}, 'scaled'), (_Custom, {}, 'shifted'),
                          (KS, dict(Nx=32), 'time_step'), (KS, dict(Nx=32), 'get_energy'),
                          (KS, dict(Nx=32), 'reset_derived'), (KS, dict(Nx=32), 'copy'),
                          (KS2D, dict(Nx=16, Ny=16), 'time_step'), (KS2D, dict(Nx=16, Ny=16), 'get_observables'),
                          (KS2D, dict(Nx=16, Ny=16), '_lhat')],
                         ids=['method', 'overridden_method', 'staticmethod', 'inherited_staticmethod', 'classmethod',
                              'singledispatchmethod', 'partialmethod',
                              'KS-time_step', 'KS-get_energy', 'KS-reset_derived', 'KS-inherited',
                              'KS2D-time_step', 'KS2D-get_observables', 'KS2D-_lhat'])
def test_keyword_that_names_a_method_is_ignored(attribution, cls, kwargs, method):
    """A keyword that names a method would replace it, e.g. Lorenz63(copy=1) and then model.copy() fails. KS and KS2D
    set the keywords that name their own class attributes before Model.__init__, which must skip their methods too."""
    with pytest.warns(UserWarning, match=re.escape(f"{cls.__name__}: keyword(s) ['{method}'] name methods")) as rec:
        line = _next_line()
        model = cls(**kwargs, **{method: 1})
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert method not in vars(model) and getattr(model, method) != 1
    if method in ('scaled', 'shifted'):     # the method still works
        assert getattr(model, method)(3) == {'scaled': 6, 'shifted': 4}[method]
    model.time_integrate(Nt=2)      # the forecast of KS calls time_step
    model.copy().close()
    model.close()


def test_keywords_of_both_kinds_warn_separately(attribution):
    with pytest.warns(UserWarning) as rec:
        line = _next_line()
        Lorenz63(copy=1, sigam=5.).close()
    messages = sorted(str(w.message) for w in _ignored(rec))
    assert messages == ["Lorenz63: keyword(s) ['copy'] name methods of the model and are ignored",
                        "Lorenz63: keyword(s) ['sigam'] are not attributes of the model and are ignored"]
    assert {(w.filename, w.lineno) for w in _ignored(rec)} == {(__file__, line)}


class _CallableValues(Lorenz63):
    """A user subclass whose class attributes hold callable values, which are not methods: they do not bind."""

    nonlinearity = np.tanh
    container = dict
    shaper = abs


@pytest.mark.parametrize('key, value', [('nonlinearity', np.sin), ('container', list), ('shaper', round)])
def test_keyword_sets_a_callable_value(key, value):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model = _CallableValues(**{key: value})
    assert getattr(model, key) is value and getattr(_CallableValues, key) is not value
    model.close()


def test_a_partial_is_a_method_where_it_binds():
    """From Python 3.14, functools.partial binds to the instance, as a function does: a keyword cannot replace it."""
    binds = hasattr(functools.partial, '__get__')

    class WithPartial(Lorenz63):
        scaled = functools.partial(max, 0)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        model = WithPartial(scaled=min)
        scaled = model.scaled           # Python 3.13 warns here that a partial will bind
    assert bool(_ignored(rec)) == binds and (scaled is min) != binds
    model.close()


def test_keyword_that_names_a_property_sets_it():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model = _Custom(gain=2.)
    assert model.gain == 2. and _Custom._gain == 1.
    model.close()


def test_rijke_tau_ensemble_does_not_warn():
    """Estimating tau re-sizes the Rijke state inside init_ensemble: no warning against re-assigning psi0."""
    model = Rijke()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model.init_ensemble(m=4, est_alpha=['tau'])
    assert model.Nc == 50 and model.psi0.shape == (model.Nphi, 1)
    assert model.hist.shape == (1, model.Nphi + 1, 4)
    model.close()


def test_typeguard_requirement_excludes_typeguard_2_and_the_releases_that_python_3_14_cannot_import():
    """typeguard 2 wraps @typechecked functions, so every warning of Model.__init__ pointed at typeguard. typeguard
    3.0 to 4.1.2 import ast.Str, which Python 3.14 removed; 4.1.3 imports and passes the suite on 3.11 and 3.14."""
    tomllib = pytest.importorskip('tomllib')  # Python 3.11+
    requirements = pytest.importorskip('packaging.requirements')
    pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    if not pyproject.is_file():
        pytest.skip('no pyproject.toml next to the tests')
    with pyproject.open('rb') as f:
        dependencies = [requirements.Requirement(d) for d in tomllib.load(f)['project']['dependencies']]
    (typeguard,) = [r for r in dependencies if r.name == 'typeguard']
    assert not any(typeguard.specifier.contains(v) for v in ('2.13.3', '3.0.0', '4.1.2'))
    assert typeguard.specifier.contains('4.1.3')


def test_typeguard_adds_no_frame_to_model_init():
    """typeguard 3 and later instrument Model.__init__ in place, so the warnings issued in it skip no typeguard line."""
    init = dynamodels.model.Model.__init__
    assert init.__code__.co_filename == dynamodels.model.__file__ and not hasattr(init, '__wrapped__')


def test_ignored_keyword_keeps_the_default():
    with pytest.warns(UserWarning, match=r"\['sigam'\] are not attributes"):
        model = Lorenz63(sigam=5.)
    assert model.sigma == Lorenz63.sigma
    model.close()


def test_known_keywords_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for model in (Lorenz63(rho=30., dt=0.01, observe_dims=[2], t_transient=1.),
                      KS(Nx=32, seed=3, Nq=4), VdP(t_transient=0.05), Rijke(case=None, beta=1.)):
            model.close()


def test_each_call_site_shows_its_warning_once(attribution):
    """The default filter shows a warning once per location: two lines show two warnings, a loop on one line one."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('default')
        Lorenz63(sigam=10.).close()
        Lorenz63(sigam=10.).close()
        for _ in range(3):
            Lorenz63(sigam=10.).close()
    assert len(_ignored(rec)) == 3


def test_model_warnings_name_the_caller(attribution):
    model = Lorenz63()
    with pytest.warns(UserWarning, match='psi0 is being re-assigned') as rec:
        line = _next_line()
        model.psi0 = np.ones(3)
    assert (rec[0].filename, rec[0].lineno) == (__file__, line)
    with pytest.warns(UserWarning, match='Cannot set est_alpha') as rec:
        line = _next_line()
        model.est_alpha = ['rho']
    assert (rec[0].filename, rec[0].lineno) == (__file__, line)
    with pytest.warns(UserWarning, match='No ensemble configuration') as rec:
        line = _next_line()
        model.visualize_state()
    assert (rec[0].filename, rec[0].lineno) == (__file__, line)
    model.close()
