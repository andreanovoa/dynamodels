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
    """Each panel gives "mean ± standard deviation", with a positive standard deviation for a negative mean."""
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
    texts = [ax.get_title(loc='left') for ax in plt.gcf().axes if '±' in ax.get_title(loc='left')]
    assert texts and all(float(t.split('±')[1]) >= 0 for t in texts)
    assert all(float(t.split('±')[0]) < 0 for t in texts)           # the means of this ensemble are negative
    plt.close('all')


def test_mean_std_text_rounds_to_the_spread():
    from dynamodels.model import _mean_std_text

    assert _mean_std_text(2574.19, 31.2) == '2574 ± 31'
    assert _mean_std_text(69.8731, 0.0213) == '69.873 ± 0.021'
    assert _mean_std_text(-1.75e-16, 2.5e-17) == '-1.75e-16 ± 2.5e-17'
    assert _mean_std_text(3., 0.) == '3 ± 0'


def test_state_distribution_caps_the_state_panels():
    """A long state shows its first eight variables, and says so; the parameters keep their panels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from dynamodels.model import plot_state_distribution
    from dynamodels.physical import Rijke

    model = Rijke()
    model.init_ensemble(m=6, std_phi=0.2, std_alpha=dict(beta=(3., 4.)))
    plot_state_distribution(model)
    fig = plt.gcf()
    assert sum(ax.axison for ax in fig.axes) == 8 + 1
    assert 'first 8 of' in fig.get_suptitle()
    plt.close('all')
