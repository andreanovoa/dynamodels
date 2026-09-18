"""Generate the timeseries figures used on the docs model pages.

Runs each physical model at its (representative) class defaults, past its own
transient, and saves the figure its own `Model.visualize_*` method already
knows how to draw -- observable timeseries for point models, a space-time
diagram for the field models. Also saves one `ntsa.characterize` panel for the
Lorenz63 case, referenced from the docs' analysis page.

Not run by CI: figures are static assets checked into `docs/img/`. Re-run by
hand after a model or its defaults change:

    python scripts/generate_docs_figures.py
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt

from dynamodels.physical import KS, KS2D, Annular, Kuznetsov, Lorenz63, Lorenz96, Rijke, VdP

IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'img')

# (slug, model class, kwargs, which Model.visualize_* method draws the figure)
MODELS = [
    ('lorenz63', Lorenz63, {}, 'visualize_observable_hist'),
    ('lorenz96', Lorenz96, {}, 'visualize_spatiotemporal_hist'),
    ('van_der_pol', VdP, {}, 'visualize_observable_hist'),
    ('rijke', Rijke, {}, 'visualize_observable_hist'),
    # ER=0.5 class default gives nu<0 (linearly stable, decays to the origin);
    # the mixed mode from the class docstring is the illustrative self-sustained case
    ('annular', Annular, dict(nu=20., c2beta=18.), 'visualize_observable_hist'),
    ('ks', KS, {}, 'visualize_spatiotemporal_hist'),
    ('ks2d', KS2D, {}, 'visualize_spatiotemporal_hist'),
    ('kuznetsov', Kuznetsov, {}, 'visualize_observable_hist'),
]


def run_and_plot(slug, cls, kwargs, method):
    model = cls(**kwargs)
    # past the transient, plus enough cycles for the zoomed panel to be legible
    Nt = int((model.t_transient + 15 * model.t_CR) / model.dt)
    psi, t = model.time_integrate(Nt=Nt)
    model.update_history(psi, t)

    getattr(model, method)()
    out = os.path.join(IMG_DIR, f'{slug}.png')
    plt.gcf().savefig(out, dpi=140, bbox_inches='tight')
    plt.close('all')
    model.close()
    print(f'wrote {out}')


def characterize_lorenz63():
    from ntsa.characterize import characterize
    out_pdf = os.path.join(IMG_DIR, 'ntsa_characterize_lorenz63.pdf')
    # t_run shorter than the default 100*t_CR: the 3-D delay-portrait/MDS scatters
    # are what's slow to rasterize, not the integration, and 80 time units (~18
    # Lyapunov times) is already a representative attractor for a docs figure.
    characterize(Lorenz63(), labels='Lorenz63 (rho=28)', t_run=80., pdf_name=out_pdf)
    # characterize() saves the row-grid page as a same-name PNG too
    print(f'wrote {out_pdf.replace(".pdf", ".png")}')


if __name__ == '__main__':
    os.makedirs(IMG_DIR, exist_ok=True)
    for args in MODELS:
        run_and_plot(*args)
    characterize_lorenz63()
