# Smurves

#### The triple-random constrainable smooth curve generator for function perturbation
<img src="/logo.png" alt="logo" width="200px"/>

Smurves is a tool for random smooth curve generation with a left-hand convergence point that allows for several constraints to be put on the generation process. If offers a more constrainable alternative to using, for example, Gaussian processes for that purpose. The approach is based on Newtonian [projectile motion](https://en.wikipedia.org/wiki/Projectile_motion) and takes its inspiration from Brandon Sanderson's book series [The Stormlight Archive](https://brandonsanderson.com/books/the-stormlight-archive/) and the books' magic system called "surgebinding". Specifically, the part that Sanderson calls a "basic lashing" deals with the change of direction and magnitude of gravity for an object.

With that idea in mind, Smurves generates smooth curves by randomly sampling the gravitational force, as well as both points and number of changes in gravity direction, for one projectile per curve path and with velocity and angle being retained through gravitational direction switches. 

### Installation

Scadda can be installed via [PyPI](https://pypi.org), with a single command in the terminal:

```
pip install scadda
```

Alternatively, the file `scadda.py` can be downloaded from the folder `scadda` in this repository and used locally by placing the file into the working directory for a given project. An installation via the terminal is, however, highly recommended, as the installation process will check for the package requirements and automatically update or install any missing dependencies, thus sparing the user the effort of troubleshooting and installing them themselves.

### Quickstart guide

In addition to the number of curves and interval constraints for both x-axis and y-axis beyond which the curves shouldn't stray, the tool requires the number of equally spaced measurement points, the maximum number of gradient changes in the curves, and a left-hand convergence point with the same x-axis value as the left side of the `x_interval parameter`. Three optional parameters include the choice of a logarithmic scale for the x-axis, the use of a mirrored truncated Gaussian distribution instead of the default uniform distribution to sample gravitational forces from in order to let most curves not deviate too far, and the placing of a threshold point before which no deviation from the convergence point's x-axis value should take place. These parameters are described in the table further below. can be




| Variables              | Explanations                                    | Default |
|:-----------------------|:------------------------------------------------|:--------|
| n_curves               | The number of curves you want to generate       |         |
| x_interval             | The allowed x-axis interval for the curves      |         |
| y_interval             | The allowed y-axis interval for the curves      |         |
| n_measure              | The number of equally-spaced measurement points |         |
| direction_maximum      | The maximum number of allowed gradient changes  |         |
| convergence_point      | The left-side point of convergence for curves   |         |
| log_scale (optional)   | Whether measurements should be on a log-scale   | False   |
| trunc_norm (optional)  | Whether curves shouldn't stay closer to unity   | False   |
| start_force (optional) | The point of the first deviation from unity     | None    |




After the installation via [PyPI](https://pypi.org), or using the `smurves.py` file locally, the usage looks like this:

```python
from smurves import surgebinder

curves = surgebinder(n_curves = 100,
                     x_interval = [0, 5],
                     y_interval = [0, 1],
                     n_measure = 100,
                     direction_maximum = 1,
                     convergence_point = [0, 2],
                     log_scale = True,
                     trunc_norm = False,
                     start_force = 0.1)
```

