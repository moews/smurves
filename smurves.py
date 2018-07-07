"""The triple-random smooth curve generator for function perturbation.

Introduction:
-------------
Smurves is a tool for random smooth curve generation that allows for
several constraints to be put on the generation process. If offers a
more constrainable alternative to using, for example, Gaussian processes
for that purpose. The approach is based on Newtonian projectile motion
and takes its inspiration from Brandon Sanderson's book series The
Stormlight Archive and the books' magic system called "surgebinding".
Specifically, the part that Sanderson calls a "basic lashing" deals with
the change of direction and magnitude of gravity for an object.

With that idea in mind, Smurves generates smooth curves by randomly
sampling the gravitational force, as well as locations and the number of
changes in the direction of gravity, for one projectile per curve path
and making sure that the constraints are adhered to, and with velocity
and angle being retained after gravitational direction switches.

In essence, you can imagine the code's inner workings as firing a bullet
at either zero angle or a random angle, from a specified or random point
at either the left or the right side of the x-axis interval, depending
on the user preferences. At a random number of random points, gravity
gets turned upside down with a random magnitude, while the bullet
continues its flight, all subject to the constraints set by the user
for the properties of the required curves.

The motivation for this approach was to find a novel way of generating
utterly random curves with certain constraints to perturbate functions,
for example the matter power spectrum in cosmology.

Quickstart:
-----------
To start using smurves, simply use "from smurves import surgebinder" to
access the primary function. The exact requirements for the inputs are
listed in the docstring of the surgebinder() function further below.
An example for using Smurves looks like this:

    ----------------------------------------------------------------
    |  from smurves import surgebinder                             |
    |                                                              |
    |  curves = surgebinder(n_curves = 10,                         |
    |                       x_interval = [0.0, 5.0],               |
    |                       y_interval = [0.0, 2.0],               |
    |                       n_measure = 100,                       |
    |                       direction_maximum = 1,                 |
    |                       convergence_point = [0.0, 1.0])        |
    |                                                              |
    ----------------------------------------------------------------

Author:
--------
Ben Moews
Institute for Astronomy (IfA)
School of Physics & Astronomy
The University of Edinburgh
"""
# Import the necessary libraries
import warnings
import numpy as np
from scipy.stats import truncnorm
from random import uniform, sample

# Ignore irrelevant lower outputs
warnings.filterwarnings("ignore")

def surgebinder(n_curves,
                x_interval,
                y_interval,
                n_measure,
                direction_maximum,
                convergence_point = None,
                log_scale = False,
                trunc_norm = False,
                random_launch = False,
                right_convergence = False,
                change_range = None,
                start_force = None):
    """
    Generate random smooth curves while fulfilling given constraints.

    This is the primary function of Smurves, allowing access to its
    bundled functionality with a simple function call. Depending on the
    input parameters, each curve is generated with a projectile path for
    multiple random values that govern its behavior.

    Parameters:
    -----------
    n_curves : int >= 1
        The number of curves that are to be returned to the user. This
        is simply the value that indicates how many curves are needed
        for whatever goal they'll be used after being generated.

    x_interval : list with two single floats
        The x-axis interval for curves, as [left point, rigth point].
        This range indicates over which x-axis span the measurements
        for the curves should be done, i.e. the range of the curves.

    y_interval : list with two single floats
        The x-axis interval for curves, as [lower point, upper point].
        This range indicates which y-axis window curves shouldn't leave
        under any circumstances to make them still useful to the user.

    n_measure : int >= 0
        The number of equally-spaced measurement points on the x-axis
        for each curve. If the parameter 'log_scale' is set to True, the
        points will be equally-spaced only if depicted on a logarithmic
        x-axis. Otherwise, they will be equally-spaced on linear scales.

    direction_maximum : int >= 0
        The maximum number of gravity flips, i.e. direction changes.
        This value determines the upper end of the range from which a
        number of gravity direction change points is sample uniformly
        as integers, with 0 as the lower end of the sampling range.

    convergence_point : list with two single floats, defaults to None
        The point in which all curves should perfectly converge, as
        [x-axis value, y-axis value]. Normally, this refers to left-side
        convergence if the parameter 'right_convergence' isn't set to
        True. If 'convergence_point' isn't set, projectile starting
        points are sampled uniformly random from the y-axis interval,
        but the projectile will still start at a zero launch angle.

    log_scale : bool, defaults to False
        The indicator whether the measurements on the x-axis should be
        on a logarithmic scale, while retaining the behavior of a code
        calculation for a linear scale. This means that the steps will
        be equally-spaced when displayed with a logarithmic x-axis.

    trunc_norm : bool, defaults to False
        The indicator whether the random sampling of gravity magnitudes
        between 0 and the maximum allowable magnitude to not overshoot
        the y-axis interval shouldn't be done uniformly, but from a
        mirrored truncated Gaussian. This means that, on average, the
        curves will deviate less from the y-axis starting point.

    random_launch : bool, defaults to False
        The indicator whether no initial zero launch angle is necessary,
        i.e. projectiles will start at random angles sampled uniformly
        between -90 and 90 degrees for each curve separately.

    right_convergence : bool, defaults to False
        The indicator whether curves should converge on the right side
        instead of the left side. After computing the curves, their
        y-axis measurement vector will be flipped. If 'log_scale' is set
        to True and a value for 'start_force' is provided, this means
        that the 'start_force' threshold value for the first deviation
        from unity on the y-axis is calculated for left-side convergence
        before being flipped, which should be considered in the inputs.

    change_range : list with two single floats, defaults to None
        The x-axis percentiles below and above which no gravity flips
        should take place to avoid extreme bends in the curves due to
        the gravitational magnitude being sampled up to the maximum
        allowable force to hit the upper limit of the y-axis interval,
        as [lower percentile, upper percentile]. The default behavior if
        the parameter isn't set is to use the 10th and 90th percentile.

    start_force : float, defaults to None
        The x-axis point before which no y-axis deviation with regard to
        the projectile's starting point should happen. This is useful
        if a function perturbation should only happen after a certain
        point, which can be specified by setting this parameter.

    Returns:
    --------
    curves: list
        The generated curves in a list, with one list element per curve.
        Each list elemenet contains two rows, the first for the x-axis
        measurement points and the second for the y-axis measurements.

    Attributes:
    -----------
    None
    """
    print("Generating random curves ...\n")
    # If no change range is given, set limits to 10% and 90%
    if change_range == None:
        change_range = [0.1, 0.9]
    # Check if one singular convergence point was requested
    if convergence_point == None:
        convergence_flag = True
    else:
        convergence_flag = False
    # Calculate both the step size and measurement locations
    difference = np.diff([x_interval[0], x_interval[1]])
    step_size = np.divide(difference, n_measure - 1)
    steps = [x_interval[0] + np.multiply(i, step_size)
             for i in range(0, n_measure)]
    # Initialize an empty list for storing the curves later
    curves = []
    # Remember the initially user-requested number of curves
    orig_request = n_curves
    error_factor = np.multiply(2, direction_maximum)
    if random_launch == True:
        error_factor = np.multiply(error_factor, 2)
    n_curves = int(np.multiply(n_curves, np.maximum(1, error_factor)))
    if n_curves >= 10:
        # Set 10%-based printout milestones for progress updates
        iter_range = np.array_split(np.arange(0, n_curves), 10)
        print_points = [entry[-1] for entry in iter_range]
        percentage = 0
    else:
        progress_update = 0
    # Set an indicator for a requested flat state at the start
    if start_force == None:
        flat_state = False
    else:
        flat_state = True
        # If for log-scale, recalculate the flat state ending
        if log_scale == True:
            log_steps, log_step_size = logarithmic(x_interval = x_interval,
                                                    n_measure = n_measure)
            log_cut = np.min(np.where(np.asarray(log_steps) > start_force)[0])
            start_force = steps[log_cut]
        flat_value = start_force
    # Loop over the required total number of separate curves
    for curve in range(0, n_curves):
        # If no convergence point is given sample a random one
        if convergence_flag == True:
            y_convergence = uniform(y_interval[0], y_interval[1])
            convergence_point = [x_interval[0], y_convergence]
        # Reset the start force if a flat state is requested
        if flat_state == True:
            start_force = flat_value
        # Generate the random force direction change points
        allowed = range(0, np.random.randint(0, direction_maximum + 1))
        if start_force == None:
            lower_range = int(np.multiply(len(steps), change_range[0]))
        else:
            flat_change = int(np.floor(np.divide(start_force, step_size)))
            lower_range = int(np.multiply(len(steps),
                                          change_range[0]) + flat_change)
            #lower_range = int(np.ceil(start_force))
        higher_range = int(np.multiply(len(steps), change_range[1]))
        sample_range = range(lower_range, higher_range)
        sample_number = np.random.randint(0, direction_maximum + 1)
        change_points = np.sort(sample(sample_range, sample_number))
        if start_force != None:
            # Adapt the change points to allow the flat start
            if not list(change_points):
                change_points = [flat_change]
            else:
                change_points = np.hstack((flat_change, change_points))
        # Generate a random initial direction for the force
        direction = np.random.choice([-1, 1])
        # Set the particle's velocity to an arbitrary value
        velocity = 1.0
        # Set the angle to zero for a left-side convergence
        if random_launch == True:
            launch_angle = np.deg2rad(uniform(-90, 90))
        else:
            launch_angle = 0.0
        # Get the maximum force to stay within the intervals
        rest_time = np.divide(x_interval[1], velocity  )
        max_range = y_interval[np.maximum(0, direction)] - convergence_point[1]
        abs_max = np.multiply(np.negative(direction), (max_range))
        spread = np.multiply(velocity, np.sin(launch_angle)) - abs_max
        force_max = np.divide(np.multiply(2, spread), np.square(rest_time))
        # Randomly sample the force depending on preferences
        if trunc_norm == True:
            distribution = truncnorm.rvs(a = -2, b = 2)
            force = np.multiply(np.abs(distribution), np.divide(force_max, 2))
        else:
            force = uniform(0, force_max)
        # Set the convergence point as the first start point
        start_point = convergence_point
        # Initialize a curve path, start distance and counter
        curve_path = [convergence_point]
        horizontal_start = [0.0]
        counter = 0
        # Initialize the beginning as the last visited point
        last_point = convergence_point
        # Loop over change points to calculate partial curves
        for part in range(0, len(change_points) + 1):
            # Set the steps depending on the process' status
            if not list(change_points):
                partial_steps = steps[counter:len(steps)]
            else:
                partial_steps = steps[counter:change_points[0] + 1]
                counter = change_points[0]
                change_points = change_points[1:len(change_points)]
                # Sample a random force for the partial curve
                scale_factor = np.divide(len(steps), len(partial_steps))
                force_max = np.multiply(force_max, scale_factor)
                if start_force == None:
                    if trunc_norm == True:
                        distr = truncnorm.rvs(a = -2, b = 2)
                        force = np.multiply(np.abs(distr),
                                            np.divide(force_max, 2))
                    else:
                        force = uniform(0, force_max)
                else:
                    force = 0.0
                    start_force = None
            # Calculate the trajectory for the partial curve
            output = trajectory(force = force,
                                velocity = velocity,
                                direction = direction,
                                step_size = step_size,
                                start_point = start_point,
                                launch_angle = launch_angle,
                                partial_steps = partial_steps,
                                horizontal_start = horizontal_start)
            # Assign the values from the trajectory function
            partial_path, last_point, launch_angle, velocity = output[0:4]
            # Update parameters for the next loop iteration
            start_point = last_point
            direction = -direction
            force = np.negative(force)
            horizontal_start = [convergence_point[0]]
             # Get the maximum force to stay within the intervals
            rest_time = np.divide(x_interval[1] - last_point[0], velocity)
            max_range = y_interval[np.maximum(0, direction)] - last_point[1]
            abs_max = np.multiply(np.negative(direction), (max_range))
            spread = np.multiply(velocity, np.sin(launch_angle)) - abs_max
            force_max = np.divide(np.multiply(2, spread), np.square(rest_time))
            # Randomly sample the force depending on preferences
            if trunc_norm == True:
                distr = truncnorm.rvs(a = -2, b = 2)
                force = np.multiply(np.abs(distr), np.divide(force_max, 2))
            else:
                force = uniform(0, force_max)
            # Convert the partial path into a congestible format
            partial_path = np.asarray(partial_path)
            if not partial_path.ndim < 2:
                partial_path = partial_path.reshape(partial_path.shape[0],
                                                    partial_path.shape[1])
                curve_path = np.vstack((curve_path, partial_path))
        # Append the computed curve to the complete set of curves
        append_point = np.asarray(last_point).T
        append_path = curve_path[1:len(curve_path), :]
        curve_path = np.vstack((append_path, append_point))
        curves.append(curve_path)
        # Print progress updates to inform about remaining time
        if n_curves >= 10:
            if curve == print_points[0]:
                percentage = percentage + 10
                print("%d %%" % percentage)
                print_points = print_points[1:len(iter_range)]
        else:
            progress_update = progress_update + 1
            if np.mod(progress_update, 2) == 0:
                print("%d curves generated" % np.divide(progress_update, 2))
    # Transform to log-scale measurements if required by the user
    if log_scale == True:
        steps, step_size = logarithmic(x_interval = x_interval,
                                       n_measure = n_measure)
        # Save the log-scale measurement points into the curves
        for i in range(0, len(curves)):
            for j in range(0, len(curves[i])):
                curves[i][j][0] = steps[j]
    # Remove curves with trajectories beyond the allowed range
    remove = []
    for i in range(0, len(curves)):
        if (any(curves[i][:, 1] < y_interval[0])
            or any(curves[i][:, 1] > y_interval[1])):
            remove.append(i)
    for index in sorted(remove, reverse = True):
        del curves[index]
    # If right-side convergence is requested, flip the values
    if right_convergence == True:
        for i in range(0, len(curves)):
            curves[i][:, 1] = np.flip(curves[i][:, 1])
    # Cut the dataset to the user-requested number of curves
    curves = curves[0:orig_request]
    print("\nComplete, returning your curves!")
    # Return the list of random curves as the function output
    return curves

def logarithmic(x_interval,
                n_measure):
    """
    Convert measurement points and step size to the logarithmic scale.

    This function converts the x-axis interval to a logarithmic scale
    by adjusting the measurement points and the step size used for the
    projectile trajectorie calculations. As a result, the step size
    won't be equally-spaced on a linear scale anymore, but will be so
    when depicted on a logarithmic scale. This automatically takes care
    of the curves' use for functions depicted on a logarithmic scale.

    Parameters:
    -----------
    x_interval : list with two single floats
        The x-axis interval for curves, as [left point, rigth point].
        This range indicates over which x-axis span the measurements
        for the curves should be done, i.e. the range of the curves.

    n_measure : int >= 0
        The number of equally-spaced measurement points on the x-axis
        for each curve. If the parameter 'log_scale' is set to True, the
        points will be equally-spaced only if depicted on a logarithmic
        x-axis. Otherwise, they will be equally-spaced on linear scales.

    Returns:
    --------
    steps : array-like
        The x-axis measurement points that are equally-spaced when they
        are depicted on a logarithmic scale. This ensures the same
        curve behavior for the linear and logarithmic curve generation.

    step_size : float
        The step size for the initial curve computation. This isn't
        relevant for the final adjustment to the logarithmic scale, but
        is necessary for calculation the projectile trajectories.

    Attributes:
    -----------
    None
    """
    # Transform the x-axis interval to the logarithmic scale
    log_interval = np.log10(x_interval)
    # Get the difference between starting and ending points
    difference = np.diff([log_interval[0], log_interval[1]])
    # Compute the new step size over the logarithmic scale
    step_size = np.divide(difference, n_measure - 1)
    # Calculate the steps with the new logarithmic step size
    steps = [np.power(10, log_interval[0] + np.multiply(i, step_size))
                 for i in range(0, n_measure)]
    # Return the steps and step size as the function output
    return steps, step_size

def trajectory(force,
               velocity,
               direction,
               step_size,
               start_point,
               launch_angle,
               partial_steps,
               horizontal_start):
    """
    Generate a partial trajectory of a particle with constant force.

    This function calculates the trajectory of a projectile for its
    given parameters, and for the partial x-axis interval over which
    the trajectory has to be calculated. The primary inputs for this
    computation are the gravitational magnitude, the initial velocity,
    the starting point, the launch angle, and the gravity direction.

    Parameters:
    -----------
    force : float
        The constant gravitational force applied to the projectile over
        the partial trajectory calculated by this function, leading to
        a smooth curve behavior along the x-axis measurement points.

    velocity : float
        The initial velocity with which the partial trajectory for the
        given projectile's path is being calculated. While this value
        remains the same, the final velocity at the end of the partial
        path that is among the returned values is calculated separately.

    direction : int from the set {-1, 1}
        The direction of gravitational influence for the calculated
        partial trajectory, i.e. an either positive or negative value
        to indicate in which direction a vertical force is applied.

    step_size : float
        The step size for the x-axis measurement points at which the
        location of the projectile along the y-axis are to be measured.
        Like the gravitational force, this value remains constant.

    start_point : list with two single floats
        The starting point of the trajectoty calculation, i.e. the
        point from which the projectile starts its flight with a given
        velocity, gravitational force and launch angle.

    launch_angle : float
        The launch angle with which the projectile for the partial path
        calculation starts its path, i.e. the firing angle from which
        the projectile is accelerated to build a curve path.

    partial_steps : list of single floats
        The x-axis measurement points at which the location of the
        projectile along the y-axis are to be measured for the partial
        trajectory for the given part of the curve calculation.

    horizontal_start : float
        The correction value that ensures that the initial horizontal
        displacement used for the calculation is suitable for the
        computations at hand, with the first partial path using a
        value of 0.0 and the subsequent partial paths, if existent,
        use a value equal to the x-axis value of the convergence point.

    Returns:
    --------
    points : list of float tuples
        The x-axis and y-axis values for each given x-axis point of
        measurement that is provided to the function, as a list with
        list elements of the form [x-axis value, y-axis value].

    last_point : list of two single floats
        The x-axis and y-axis value of the particle's final location at
        the end of the calculation, as [x-axis value, y-axis value]. If
        existent, this point will be the starting point for the the next
        partial path of the curve so that they blend into one curve.

    impact_angle : float
        The impact angle of the projectile when reaching the right-side
        end of the x-axis interval specific to the partial path that is
        being calculated with this function. If existent, the following
        partial path will be calculated with this angle as its launch
        angle for smooth curve behavior over the curve's evolution.

    velocity : float
        The final velocity of the projectile at the end of the partial
        path that is computed with this function. If existent, the next
        partial path of the curve will start with that velocity for
        smooth curve behavior over the curve's evolution.

    Attributes:
    -----------
    None
    """
    # Initialize the horizontal displacement with the input
    horizontal_displacement = [horizontal_start]
    # Calculate the first horizontal and vertical velocities
    horizontal_velocity = np.multiply(velocity, np.cos(launch_angle))
    vertical_velocity = np.multiply(velocity, np.sin(launch_angle))
    # Save the initial velocity for consecutive calculations
    start_velocity = velocity
    # Initialize a list for storing the measurement points
    points = [start_point]
    # Loop over the number of steps minus the initial step
    for i in range(1, len(partial_steps)):
        # Get the horizontal distance, displacement and time
        distance = partial_steps[i]
        horizontal_displacement = horizontal_displacement + step_size
        time = np.divide(horizontal_displacement, horizontal_velocity)
        # Calculate the vertical velocity and displacement
        interim = np.multiply(start_velocity, np.sin(launch_angle))
        vertical_velocity = interim - np.multiply(force, time)
        velocity_part = np.multiply(start_velocity,
                                    np.multiply(np.sin(launch_angle), time))
        force_part = np.multiply(0.5, np.multiply(force, np.square(time)))
        vertical_displacement = np.negative(velocity_part - force_part)
        # Calculate the total velocity at the given point
        velocity = np.sqrt(np.square(horizontal_velocity)
                           + np.square(vertical_velocity))
        # Append the current measurement point to the list
        direction_displacement = np.multiply(direction, vertical_displacement)
        points.append([distance, start_point[1] + direction_displacement])
    # Calculate both the final velocity and impact angle
    final_velocity = np.sqrt(np.square(horizontal_velocity)
                             + np.square(vertical_velocity))
    impact_angle = np.arctan(np.divide(np.negative(vertical_velocity),
                             horizontal_velocity))
    # Separate the last measurement point from the points
    last_point = points[-1]
    points = points[0:-1]
    # Return the points, last point, angle and velocity
    return points, last_point, impact_angle, velocity
