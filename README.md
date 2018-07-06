# Smurves

#### The triple-random constrainable smooth curve generator for function perturbation
<img src="/logo.png" alt="logo" width="200px"/>

Smurves is a tool for random smooth curve generation with a left-hand convergence point that allows for several constraints to be put on the generation process. If offers a more constrainable alternative to using, for example, Gaussian processes for that purpose. The approach is based on Newtonian [projectile motion](https://en.wikipedia.org/wiki/Projectile_motion) and takes its inspiration from Brandon Sanderson's book series [The Stormlight Archive](https://brandonsanderson.com/books/the-stormlight-archive/) and the books' magic system called "surgebinding". Specifically, the part that Sanderson calls a "basic lashing" deals with the change of direction and magnitude of gravity for an object.

With that idea in mind, Smurves generates smooth curves by randomly sampling the gravitational force, as well as both points and number of changes in gravity direction, for one projectile per curve path and with velocity and angle being retained through gravitational direction switches. In addition to the number of curves and interval constraints for both x-axis and y-axis beyond which the curves shouldn't stray, the tool requires the number of equally spaced measurement points, the maximum number of gradient changes in the curves, and a left-hand convergence point with the same x-axis value as the left side of the x_interval parameter. These parameters, as well as three optional ones, are described in the table further below.

### Installation

Scadda can be installed via [PyPI](https://pypi.org), with a single command in the terminal:

```
pip install scadda
```

Alternatively, the file `scadda.py` can be downloaded from the folder `scadda` in this repository and used locally by placing the file into the working directory for a given project. An installation via the terminal is, however, highly recommended, as the installation process will check for the package requirements and automatically update or install any missing dependencies, thus sparing the user the effort of troubleshooting and installing them themselves.

### Quickstart guide

SCADDA requires the user to provide spatial data (`s_data`) as a Nx2 array for N data points, with longitudes in the first and latitudes in the second column, as well as the same number of time series per spatial data point (`t_data`) as an NxM array with M as the length of the time series. The spatial (`s_limit`) and temporal (`t_limit`) maximal distances for points to be considered part of the same cluster, as well as the steepness for the logistic function used for the distance re-scaling (`steepness`) and the mininum number of neighbors required for a cluster (`minimum_neighbors`), also have to be provided. In addition, the window size for the [Paliwal adjustment window](https://ieeexplore.ieee.org/document/1171506/) can be set (`window_param`), but is optional and will default to a data-dependent rule-of-thumb calculation.

After the installation via [PyPI](https://pypi.org), or using the `scadda.py` file locally, the usage looks like this:

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

