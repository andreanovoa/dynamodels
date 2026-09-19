# Tutorials of the thermoacoustic models

Executed notebooks, rendered to HTML (outputs, figures and animations included)
on every docs build; the GitHub column opens the `.ipynb` itself.

| Notebook | What it covers | |
| --- | --- | --- |
| [The Rijke tube](tutorials/tutorial_rijke.html) | The longitudinal thermoacoustic low-order model: the Galerkin and Chebyshev discretizations, the acoustic modes $\eta_j$ and $\mu_j$, the advection equation that carries the delayed flame velocity, the pressure at the microphone locations, and an animation of the pressure field along the tube. | [GitHub](https://github.com/andreanovoa/dynamodels/blob/main/tutorial_rijke.ipynb) |
| [The annular combustor](tutorials/tutorial_annular.html) | The azimuthal thermoacoustic low-order model: the mode pair $(\eta_a, \eta_b)$ and its phase space, the pressure at four microphones around the annulus, and an animation of the azimuthal pressure field. | [GitHub](https://github.com/andreanovoa/dynamodels/blob/main/tutorial_annular.ipynb) |


<!-- 
To regenerate the HTML locally: `python -m nbconvert --to html tutorial_*.ipynb --output-dir docs/tutorials`. -->
