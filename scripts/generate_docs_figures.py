"""Generate the figures used on the docs model pages.

Four kinds of output, all under docs/img/:

- `<slug>.png` -- one model at a representative case, run past its own
  transient, plotted with its own `Model.visualize_*` method.
- `<slug>_<case>.png` -- one per named regime, for the models with a `case=`
  parameter (Annular, Rijke, KS2D).
- `ks2d_chaotic.gif` -- an animation of the KS2D chaotic case's field, which
  a static image cannot convey for a genuinely 2-D spatial system.
- `rijke_chaotic.gif`, `annular_mixed.gif` -- animations pairing the spatial
  pressure field (along the tube / around the annulus) with the scrolling
  microphone traces, in the style of romda's thermoacoustics tutorials
  (`20_Rijke_LOM.ipynb`, `23_TA_azimuthal_LOM.ipynb`).
- `ntsa_characterize_<slug>.png` -- one `ntsa.characterize` diagnostic row
  per model, for the "Analysing a model" section of each model page.

Not run by CI: figures are static assets checked into `docs/img/`. Re-run by
hand after a model or its defaults change:

    python scripts/generate_docs_figures.py
"""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from dynamodels.physical import KS, KS2D, Annular, Kuznetsov, Lorenz63, Lorenz96, Rijke, VdP

IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'img')

# (slug, model class, kwargs, which Model.visualize_* method draws the figure)
# Annular and Rijke are covered entirely by CASE_MODELS below -- their "main"
# figure is just the first of their named cases, so there is no separate entry
# here to avoid regenerating the same run twice under two filenames.
MODELS = [
    ('lorenz63', Lorenz63, {}, 'visualize_observable_hist'),
    ('lorenz96', Lorenz96, {}, 'visualize_spatiotemporal_hist'),
    ('van_der_pol', VdP, {}, 'visualize_observable_hist'),
    ('ks', KS, {}, 'visualize_spatiotemporal_hist'),
    ('ks2d', KS2D, {}, 'visualize_spatiotemporal_hist'),
    ('kuznetsov', Kuznetsov, {}, 'visualize_observable_hist'),
]

# extra per-model regime figures, one file per named case: <slug>_<case>.png
CASE_MODELS = [
    ('annular', Annular, ['spinning', 'standing', 'mixed'], 'visualize_observable_hist'),
    ('rijke', Rijke, ['limit_cycle', 'frequency_locked', 'chaotic', 'relaminarized'],
     'visualize_observable_hist'),
]

# (slug, model instance, label, t_run, extra kwargs) fed to ntsa.characterize;
# t_run is chosen so that t_run/dt stays around 4000 samples -- long enough to
# be representative, short enough that the 3-D delay-portrait/MDS line
# rendering (the slow step, not the integration) finishes in about a minute
# per model.
NTSA_CASES = [
    ('lorenz63', Lorenz63(), 'Lorenz63 (rho=28)', 80., {}),
    ('lorenz96', Lorenz96(), 'Lorenz96 (F=8, Nx=40)', 40., {}),
    ('van_der_pol', VdP(), 'VdP (beta=70, zeta=60, kappa=4)', 0.4, {}),
    # spectrum=False: Rijke's 30-D state makes the finite-difference Jacobian
    # (needed by the full Benettin QR spectrum) far slower than the Jacobian-free
    # leading_lyapunov perturbation-growth estimate used everywhere else here
    ('rijke', Rijke(case='limit_cycle'), 'Rijke (beta=4, limit cycle)', 0.4,
     dict(spectrum=False)),
    ('annular', Annular(case='mixed'), 'Annular (mixed mode)', 0.08, {}),
    ('ks', KS(), 'KS (nu=0.08)', 1000., {}),
    ('ks2d', KS2D(case='chaotic'), 'KS2D (chaotic)', 40., {}),
    ('kuznetsov', Kuznetsov(), 'Kuznetsov (quasiperiodic)', 40., {}),
]


def run_and_plot(slug, cls, kwargs, method, suffix=''):
    model = cls(**kwargs)
    # past the transient, plus enough cycles for the zoomed panel to be legible
    Nt = int((model.t_transient + 15 * model.t_CR) / model.dt)
    psi, t = model.time_integrate(Nt=Nt)
    model.update_history(psi, t)

    getattr(model, method)()
    out = os.path.join(IMG_DIR, f'{slug}{suffix}.png')
    plt.gcf().savefig(out, dpi=140, bbox_inches='tight')
    plt.close('all')
    model.close()
    print(f'wrote {out}')


def make_ks2d_gif(n_frames=100, stride=15):
    """Animate the KS2D chaotic case: a static image cannot show the field's
    continuous evolution for a genuinely 2-D spatial system the way it can
    for the 1-D KS or the lattice models."""
    model = KS2D(case='chaotic')
    Nt_transient = int(model.t_transient / model.dt)
    psi, t = model.time_integrate(Nt=Nt_transient)
    model.update_history(psi[[-1]], t=t[[-1]], reset=True)

    psi, t = model.time_integrate(Nt=n_frames * stride)
    model.update_history(psi, t)
    y = model.get_observable_hist(loc='all')[::stride, :, 0]
    u = y.reshape(-1, model.Nx, model.Ny)
    t_frames = model.hist_t[::stride]
    lim = np.max(np.abs(u))
    model.close()

    fig, ax = plt.subplots(figsize=(4, 4), layout='constrained')
    im = ax.imshow(u[0].T, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim,
                   extent=(0, 2 * model.l, 0, 2 * model.l))
    ax.set(xlabel='$x$', ylabel='$y$')
    title = ax.set_title(f'$t={t_frames[0]:.1f}$')

    def update(i):
        im.set_data(u[i].T)
        title.set_text(f'$t={t_frames[i]:.1f}$')
        return im, title

    anim = FuncAnimation(fig, update, frames=len(u))
    out = os.path.join(IMG_DIR, 'ks2d_chaotic.gif')
    anim.save(out, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f'wrote {out}')


def make_rijke_gif(n_frames=80, stride=15):
    """Animate the Rijke chaotic case: the acoustic pressure field along the
    tube next to the scrolling microphone traces, as in romda's
    `20_Rijke_LOM.ipynb` tutorial."""
    model = Rijke(case='chaotic')
    Nt_transient = int(model.t_transient / model.dt)
    psi, t = model.time_integrate(Nt=Nt_transient)
    model.update_history(psi[[-1]], t=t[[-1]], reset=True)

    psi, t = model.time_integrate(Nt=n_frames * stride)
    model.update_history(psi, t)
    p_field = model.get_observable_hist(loc='all')[::stride, :, 0]
    p_mic = model.get_observable_hist()[::stride, :, 0]
    t_frames = model.hist_t[::stride]
    lim_field = np.max(np.abs(p_field))
    lim_mic = np.max(np.abs(p_mic))
    model.close()

    fig, (ax_field, ax_mic) = plt.subplots(
        2, 1, figsize=(5.5, 4), layout='constrained', height_ratios=[1, 2])
    im = ax_field.imshow(np.tile(p_field[0], (2, 1)), aspect='auto', origin='lower',
                          cmap='RdBu_r', vmin=-lim_field, vmax=lim_field, extent=(0, model.L, 0, 1))
    ax_field.axvline(model.xf, color='k', ls='--', lw=1)
    ax_field.set(xlabel='$x$', yticks=[], title=f'Acoustic pressure, $t={t_frames[0]:.3f}$')

    lines = [ax_mic.plot([], [], lw=1.5, label=lbl)[0] for lbl in model.obs_labels]
    ax_mic.set(xlabel='$t$', ylabel="$p'$", xlim=(t_frames[0], t_frames[-1]),
               ylim=(-lim_mic, lim_mic))
    ax_mic.legend(loc='upper right', fontsize=7, ncol=2)

    def update(i):
        im.set_data(np.tile(p_field[i], (2, 1)))
        ax_field.set_title(f'Acoustic pressure, $t={t_frames[i]:.3f}$')
        for q, line in enumerate(lines):
            line.set_data(t_frames[:i + 1], p_mic[:i + 1, q])
        return [im, *lines]

    anim = FuncAnimation(fig, update, frames=len(t_frames))
    out = os.path.join(IMG_DIR, 'rijke_chaotic.gif')
    anim.save(out, writer=PillowWriter(fps=12), dpi=70)
    plt.close(fig)
    print(f'wrote {out}')


def make_annular_gif(n_frames=80, stride=5):
    """Animate the Annular mixed-mode case: the azimuthal pressure field
    around the annulus next to the scrolling microphone traces, as in
    romda's `23_TA_azimuthal_LOM.ipynb` tutorial."""
    model = Annular(case='mixed')
    Nt_transient = int(model.t_transient / model.dt)
    psi, t = model.time_integrate(Nt=Nt_transient)
    model.update_history(psi[[-1]], t=t[[-1]], reset=True)

    psi, t = model.time_integrate(Nt=n_frames * stride)
    model.update_history(psi, t)
    angles = np.linspace(0, 2 * np.pi, 200)
    p_theta = model.get_observable_hist(loc=angles)[::stride, :, 0]
    p_mic = model.get_observable_hist()[::stride, :, 0]
    t_frames = model.hist_t[::stride]
    lim = np.max(np.abs(p_theta))
    lim_mic = np.max(np.abs(p_mic))
    model.close()

    radius = np.array([0.5, 1.])
    theta, r = np.meshgrid(angles, radius)

    fig = plt.figure(figsize=(7.5, 3.5), layout='constrained')
    ax_field = fig.add_subplot(1, 2, 1, projection='polar')
    ax_mic = fig.add_subplot(1, 2, 2)

    mesh = ax_field.pcolormesh(theta, r, np.tile(p_theta[0], (2, 1)), shading='auto',
                                cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax_field.set(yticklabels=[], theta_zero_location='N', rgrids=[], thetagrids=[])
    title = ax_field.set_title(f'Acoustic pressure, $t={t_frames[0]:.4f}$')

    lines = [ax_mic.plot([], [], lw=1.5, label=lbl)[0] for lbl in model.obs_labels]
    ax_mic.set(xlabel='$t$', ylabel="$p'$", xlim=(t_frames[0], t_frames[-1]),
               ylim=(-lim_mic, lim_mic))
    ax_mic.legend(loc='upper right', fontsize=7)

    def update(i):
        mesh.set_array(np.tile(p_theta[i], (2, 1)).ravel())
        title.set_text(f'Acoustic pressure, $t={t_frames[i]:.4f}$')
        for q, line in enumerate(lines):
            line.set_data(t_frames[:i + 1], p_mic[:i + 1, q])
        return [mesh, title, *lines]

    anim = FuncAnimation(fig, update, frames=len(t_frames))
    out = os.path.join(IMG_DIR, 'annular_mixed.gif')
    anim.save(out, writer=PillowWriter(fps=12), dpi=70)
    plt.close(fig)
    print(f'wrote {out}')


def characterize_model(slug, model, label, t_run, extra=None):
    from ntsa.characterize import characterize
    out_pdf = os.path.join(IMG_DIR, f'ntsa_characterize_{slug}.pdf')
    characterize(model, labels=label, t_run=t_run, pdf_name=out_pdf, **(extra or {}))
    # characterize() saves the row-grid page as a same-name PNG too
    print(f'wrote {out_pdf.replace(".pdf", ".png")}')


if __name__ == '__main__':
    os.makedirs(IMG_DIR, exist_ok=True)

    for args in MODELS:
        run_and_plot(*args)

    for slug, cls, cases, method in CASE_MODELS:
        for case in cases:
            run_and_plot(slug, cls, dict(case=case), method, suffix=f'_{case}')

    make_ks2d_gif()
    make_rijke_gif()
    make_annular_gif()

    for slug, model, label, t_run, extra in NTSA_CASES:
        characterize_model(slug, model, label, t_run, extra)
