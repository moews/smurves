# Smurves

#### The triple-random smooth curve generator for function perturbation
<img src="/logo.png" alt="logo" width="200px"/>

Smurves is a tool for random smooth curve generation that allows for several constraints to be put on the generation process. If offers a more constrainable alternative to using, for example, Gaussian processes for that purpose. The approach is based on Newtonian [projectile motion](https://en.wikipedia.org/wiki/Projectile_motion) and takes its inspiration from Brandon Sanderson's book series [The Stormlight Archive](https://brandonsanderson.com/books/the-stormlight-archive/) and the books' magic system called "surgebinding". Specifically, the part that Sanderson calls a "basic lashing" deals with the change of direction and magnitude of gravity for an object, which provided the initial concept.

With that idea in mind, Smurves generates smooth curves by randomly sampling the gravitational force, as well as locations and the number of changes in the direction of gravity, for one projectile per curve path and making sure that the constraints are adhered to, and with velocity and angle being retained after gravitational direction switches.

In essence, you can imagine the code's inner workings as firing a bullet at either zero angle or a random angle, from a specified or random point at either the left or the right side of the x-axis interval, depending on the user preferences. At a random number of random points, gravity gets turned upside down with a random magnitude, while the bullet continues its flight, all subject to the constraints set by the user for the properties of the required curves.

The motivation for this approach was to find a novel way of generating utterly random curves with certain constraints to perturbate functions, for example the [matter power spectrum](https://en.wikipedia.org/wiki/Matter_power_spectrum) in cosmology.

### Installation

Smurves can be installed via [PyPI](https://pypi.org), with a single command in the terminal:

```
pip install smurves
```

Alternatively, the file `smurves.py` can be downloaded from the folder `smurves` in this repository and used locally by placing the file into the working directory for a given project. An installation via the terminal is, however, highly recommended, as the installation process will check for the package requirements and automatically update or install any missing dependencies, thus sparing the user the effort of troubleshooting and installing them themselves.

### Quickstart guide

The descriptions and example usage below provide a quick tutorial on Smurves. In addition, the `examples.ipynb` Jupyter Notebook in the `examples` folder in this repository show the use of the tool for various constraints and with explanations for each parameter set, and with the code necessary to plot the curves.

In addition to the number of curves and interval constraints for both x-axis and y-axis beyond which the curves shouldn't stray, the tool requires the number of measurement points amd the maximum number of directional changes per curve.

Six optional parameters include the placement of a point in which the curves should converge, the choice of a logarithmic scale for the x-axis, the choice to launch the curve trajectories at random instead of zero angles, the choice to let the curves converge on the right instead of the left side if a convergence point is provided, the percentiles along the x-axis before and which no directional gravity changes should be implemented, and the placing of a threshold point before which no deviation from the convergence point's x-axis value should take place. These parameters are described in the table below.

<br></br>

| Variables                    | Explanations                                    | Default    |
|:-----------------------------|:------------------------------------------------|:-----------|
| n_curves                     | The number of curves you want to generate       |            |
| x_interval                   | The allowed x-axis interval for the curves      |            |
| y_interval                   | The allowed y-axis interval for the curves      |            |
| n_measure                    | The number of equally-spaced measurement points |            |
| direction_maximum            | The maximum number of allowed gradient changes  |            |
| convergence_point (optional) | The left-side point of convergence for curves   |            |
| log_scale (optional)         | Whether measurements should be on a log-scale   | False      |
| random_launch (optional)     | Whether the first launch angle should be random | False      |
| right_convergence (optional) | Whether convergence should be on the right side | False      |
| change_range (optional)      | The x-axis percentiles before and after which no <br> gradient changes should take place for curves | [0.1, 0.9] |
| change_spacing (optional)    | The minimum space in measurement points between <br> the different points of a gravitational force change | None |
|change_ratio (optional)       | The multiplier for the last partial trajectory to get <br> the upper limit for the next partial trajectory's force | None |
| start_force (optional)       | The point of the first deviation from unity     | None       |

<br></br>

After the installation via [PyPI](https://pypi.org), or using the `smurves.py` file locally, the usage looks like this:

```python
from smurves import surgebinder

curves = surgebinder(n_curves = 10,
                     x_interval = [0.001, 10.0],
                     y_interval = [0.0, 5.0],
                     n_measure = 100,
                     direction_maximum = 3,
                     convergence_point = [0.001, 1.0],
                     log_scale = True,
                     change_range = [0.2, 0.8],
                     start_force = 0.01)
```

Note that if we want a logarithmic scale, the x-axis interval, as well as the `start_force` parameter to enforce no deviation before that value, have to provide powers of ten, e.g. 0.1, 10 or 1000. Given that we chose a logarithmic scale and no deviations before x = 0.01, a set generated with the above parameters can, for example, look like this:

<img src="/example.png" alt="logo" width="600px"/>
