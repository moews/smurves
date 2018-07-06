# Smurves

#### The triple-random constrainable smooth curve generator for function perturbation
<img src="/logo.png" alt="logo" width="200px"/>

Smurves is a tool for random smooth curve generation with a left-hand convergence point that allows for several constraints to be put on the generation process. If offers a more constrainable alternative to using, for example, Gaussian processes for that purpose. The approach is based on Newtonian projectile motion and takes its inspiration from Brandon Sanderson's book series _The Stormlight Archive_ and the books' magic system called "surgebinding". Specifically, the part that Sanderson calls a "basic lashing" deals with the change of direction and magnitude of gravity for an object.

SCADDA is a tool for spatio-temporal clustering with density-based distance re-scaling. Its core based on the [ST-DBSCAN](https://dl.acm.org/citation.cfm?id=1219397) algorithm, which is a previous extension of the common [DBSCAN](https://dl.acm.org/citation.cfm?id=3001507) algorithm. Extensions that are incorporated in this new method are the re-scaling of the computed distance matrix with kernel density estimation over the spatial dimensions and a modulated logistic function, as well as time series distance measurements with dynamic time warping, which makes use of global constraints via the [Sakoe-Chiba band](https://ieeexplore.ieee.org/document/1163055/) modified with the [Paliwal adjustment window](https://ieeexplore.ieee.org/document/1171506/).

As a result, SCADDA is motivated by many real-world spatio-temporal clustering problems in which the spatial distribution of data points is very varied, with large numbers of data points in a few centers and sparsely distributed data points throughout the spatial dimensions. The taken approach allows for taking these geographical issues into consideration when looking for clusters. This alleviates an issue with [ST-DBSCAN](https://dl.acm.org/citation.cfm?id=1219397), which considers most data points outside of such high-density regions as outliers. It is a general-purpose software tool that can be used to cluster any spatio-temporal dataset with known latitude and longitude, as well as a time series for a variable, for each data point.

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

