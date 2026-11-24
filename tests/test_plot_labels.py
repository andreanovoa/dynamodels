"""The axis labels of the normalized histories print the reference with three significant digits."""

import numpy as np

from dynamodels.utils import normalized_alpha, normalized_time, normalized_y


def test_time_label_rounds_the_reference():
    (t,), label = normalized_time(1.1042402826855124, np.array([0., 2.2084805653710248]))
    assert label == '$t/1.1$'
    np.testing.assert_allclose(t, [0., 2.])
    assert normalized_time(1.0, np.zeros(2))[1] == '$t$'


def test_observable_label_rounds_the_reference():
    y = np.ones((4, 2, 3))
    _, labels = normalized_y(np.array([2.6666666666666665, 10.0]), ['$x$', '$y$'], y)
    assert labels == ['$x$ / $2.67$', '$y$ / $10$']


def test_parameter_label_rounds_the_reference():
    alpha = np.ones((5, 2))
    out, labels = normalized_alpha(alpha, ['beta', 'tau'], {'beta': r'$\beta$'},
                                   reference_a={'beta': 2.6666666666666665})
    assert labels == {'beta': r'$\beta$ / 2.67', 'tau': 'tau'}
    np.testing.assert_allclose(out[:, 0], 1 / 2.6666666666666665)


def test_state_histograms_print_a_positive_std():
    """The text box of each histogram gives the standard deviation, which is positive for a negative mean."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    from dynamodels.model import plot_state_distribution
    from dynamodels.physical import VdP

    model = VdP()
    model.init_ensemble(m=8, std_phi=0.5)
    model.update_history(-np.abs(model.current_state)[np.newaxis] - 1., t=np.array([model.t]), reset=True)
    plot_state_distribution(model)
    texts = [t.get_text() for ax in plt.gcf().axes for t in ax.texts if 'Std' in t.get_text()]
    assert texts and all(float(t.split('Std:')[1]) >= 0 for t in texts)
    plt.close('all')
