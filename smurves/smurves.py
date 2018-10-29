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
import sys
import warnings
import numpy as np
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
                random_launch = False,
                right_convergence = False,
                change_range = None,
                change_spacing = None,
                change_ratio = None,
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

    change_spacing : int > 0, defaults to 1
        The minimum space on the x-axis in full steps that is required
        between gravitational direction changes, with hiher values
        resulting in increased smoothness. The parameter has to be small
        enough that the provided 'n_measure' parameter divided by the
        the 'change_spacing' parameter is equal to or larger than the
        'direction_maximum' parameter, i.e. the number of measurements
        divided by the minimum x-axis spacing has to be >= the maximum
        number of direction changes so that all possibilities will fit.

    change_ratio : float > 0, defaults to None
        The value by which the gravitational force of the previous
        partial trajectory of a given curve is multiplied to get the
        upper limit of the range from which the next partial trajectory
        of the same curve is sampled. Like 'change_spacing', this
        parameter is a way to enforce further smoothness.

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
    # Check if all provided parameter inputs are valid
    check(n_curves = n_curves,
          x_interval = x_interval,
          y_interval = y_interval,
          n_measure = n_measure,
          direction_maximum = direction_maximum,
          convergence_point = convergence_point,
          log_scale = log_scale,
          random_launch = random_launch,
          right_convergence = right_convergence,
          change_range = change_range,
          change_spacing = change_spacing,
          change_ratio = change_ratio,
          start_force = start_force)
    print("Generating random curves ...\n")
    # If no change range is given, set limits to 10% and 90%
    if change_range == None:
        change_range = [0.1, 0.9]
    # If no change spacing is given, set the spacing to 1
    if change_spacing == None:
        change_spacing = 1
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
    if n_curves >= 10:
        # Set 10%-based printout milestones for progress updates
        iter_range = np.array_split(np.arange(0, n_curves), 10)
        print_points = [entry[-1] for entry in iter_range]
        perc = 0
        progress_update = 0
    else:
        progress_update = 0
        print_points = None
        perc = None
    # Set an indicator for a requested flat state at the start
    if start_force == None:
        flat_state = False
        flat_value = None
    else:
        flat_state = True
        # If for log-scale, recalculate the flat state ending
        if log_scale == True:
            log_steps = logarithmic(x_interval = x_interval,
                                    n_measure = n_measure)
            log_cut = np.min(np.where(np.asarray(log_steps) > start_force)[0])
            start_force = steps[log_cut]
        flat_value = start_force
    # Save the number of curves to be generated separately
    curve_request = n_curves
    # Initialize the number of curves already generated
    done_curves = 0
    # Generate curves with the previously set preferences
    generator_output = generator(n_curves = n_curves,
                                 curve_request = curve_request,
                                 x_interval = x_interval,
                                 y_interval = y_interval,
                                 convergence_flag = convergence_flag,
                                 convergence_point = convergence_point,
                                 flat_state = flat_state,
                                 direction_maximum = direction_maximum,
                                 steps = steps,
                                 step_size = step_size,
                                 change_range = change_range,
                                 change_spacing = change_spacing,
                                 change_ratio = change_ratio,
                                 start_force = start_force,
                                 flat_value = flat_value,
                                 log_scale = log_scale,
                                 random_launch = random_launch,
                                 print_points = print_points,
                                 perc = perc,
                                 progress_update = progress_update,
                                 done_curves = done_curves)
    # Save the curves and the progress parameters to variables
    curves = generator_output[0]
    print_points = generator_output[1]
    perc = generator_output[2]
    progress_update = generator_output[3]
    done_curves = generator_output[4]
    # If curves got deleted, generate new ones to compensate
    while len(curves) < curve_request:
        # Generate a new curve with the previously set preferences
        generator_output = generator(n_curves = 1,
                                     curve_request = curve_request,
                                     x_interval = x_interval,
                                     y_interval = y_interval,
                                     convergence_flag = convergence_flag,
                                     convergence_point = convergence_point,
                                     flat_state = flat_state,
                                     direction_maximum = direction_maximum,
                                     steps = steps,
                                     step_size = step_size,
                                     change_range = change_range,
                                     change_spacing = change_spacing,
                                     change_ratio = change_ratio,
                                     start_force = start_force,
                                     flat_value = flat_value,
                                     log_scale = log_scale,
                                     random_launch = random_launch,
                                     print_points = print_points,
                                     perc = perc,
                                     progress_update = progress_update,
                                     done_curves = done_curves)
        # Save the curves and the progress parameters to variables
        new_curves = generator_output[0]
        print_points = generator_output[1]
        perc = generator_output[2]
        progress_update = generator_output[3]
        done_curves = generator_output[4]
        # Add the newly generate curves to the full lit of curves
        if len(new_curves) > 0:
            for i in range(0, len(new_curves)):
                curves.append(new_curves[i])
    print("\nPreparing the final output ...")
    # Transform to log-scale measurements if required by the user
    if log_scale == True:
        steps = logarithmic(x_interval = x_interval,
                            n_measure = n_measure)
        # Save the log-scale measurement points into the curves
        for i in range(0, len(curves)):
            for j in range(0, len(curves[i])):
                curves[i][j][0] = steps[j]
    # If right-side convergence is requested, flip the values
    if right_convergence == True:
        for i in range(0, len(curves)):
            curves[i][:, 1] = curves[i][:, 1][::-1]
    # Cut the dataset to the user-requested number of curves
    #curves = curves[0:orig_request]
    print("\nComplete, returning your curves!")
    # Return the list of random curves as the function output
    return curves

def deletion(curves,
             y_interval,
             n_curves):
    """
    Delete curves who overshoot outside of the set y-axis interval.

    This function checks curves against the y-axis interval set by the
    user and deletes them from the list of curves if their trajectory
    leads outside of that interval. While the gravitational force
    sampling takes care of staying within the y-axis interval with
    regard to the final point of each partial path, forward calculation
    and eventual dismissal of the curves would be too costly and is thus
    substituted with deletion and, subsequently, the generation of just
    the number of additional valid curves that are necessary.

    Parameters:
    -----------
     curves: list
        The generated curves in a list, with one list element per curve.
        Each list elemenet contains two rows, the first for the x-axis
        measurement points and the second for the y-axis measurements.

    y_interval : list
        The x-axis interval for curves, as [lower point, upper point].
        This range indicates which y-axis window curves shouldn't leave
        under any circumstances to make them still useful to the user.

    n_curves : int
        The number of curves that are to be returned to the user. This
        is simply the value that indicates how many curves are needed
        for whatever goal they'll be used after being generated.

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
    # Remove curves with trajectories beyond the allowed range
    remove = []
    delete_flag = False
    for i in range(0, len(curves)):
        if (any(curves[i][:, 1] < y_interval[0])
            or any(curves[i][:, 1] > y_interval[1])):
            remove.append(i)
            delete_flag = True
    for index in sorted(remove, reverse = True):
        del curves[index]
    return curves, delete_flag

def generator(n_curves,
              curve_request,
              x_interval,
              y_interval,
              convergence_flag,
              convergence_point,
              flat_state,
              direction_maximum,
              steps,
              step_size,
              change_range,
              change_spacing,
              change_ratio,
              start_force,
              flat_value,
              log_scale,
              random_launch,
              print_points,
              perc,
              progress_update,
              done_curves):
    """
    Generate curves, discard them if necessary and give updates.

    This function generates curves based on its input, discards them if
    they fall outside of the required y-axis interval, and takes care of
    progress updates to provide the user with a time estimate.

    Parameters:
    -----------
    n_curves : int
        The number of curves that are to be returned to the user. This
        is simply the value that indicates how many curves are needed
        for whatever goal they'll be used after being generated.

    curve_request : int
        The initial number of curves that are returned to the user. As
        this function is called to generate single additional curves if
        the original curve generation process had to delete some curves,
        the number of curves requested has to be saved separately.

    x_interval : list with two single floats
        The x-axis interval for curves, as [left point, rigth point].
        This range indicates over which x-axis span the measurements
        for the curves should be done, i.e. the range of the curves.

    y_interval : list with two single floats
        The x-axis interval for curves, as [lower point, upper point].
        This range indicates which y-axis window curves shouldn't leave
        under any circumstances to make them still useful to the user.

    convergence_flag : bool
        The indicator whether a convergence point is required by the
        user. If true, curves should perfectly convergence in that
        exact point on either the left or the right side of the x-axis
        interval, defending on the user-specified preferences.

    convergence_point : list
        The point in which all curves should perfectly converge, as
        [x-axis value, y-axis value]. Normally, this refers to left-side
        convergence if the parameter 'right_convergence' isn't set to
        True. If 'convergence_point' isn't set, projectile starting
        points are sampled uniformly random from the y-axis interval,
        but the projectile will still start at a zero launch angle.

    flat_state : bool
        The indicator whether the parameter start_force is set to a
        value different from the default, None. If so, the curves
        shouldn't deviate on the y-axis before the value determined by
        start_force on the x-axis is reached.

    direction_maximum : int
        The maximum number of gravity flips, i.e. direction changes.
        This value determines the upper end of the range from which a
        number of gravity direction change points is sample uniformly
        as integers, with 0 as the lower end of the sampling range.

    steps : array-like
        The x-axis measurement points that are equally-spaced when they
        are depicted on a logarithmic scale. This ensures the same
        curve behavior for the linear and logarithmic curve generation.

    step_size : float
        The step size for the x-axis measurement points at which the
        location of the projectile along the y-axis are to be measured.
        Like the gravitational force, this value remains constant.

    change_range : list
        The x-axis percentiles below and above which no gravity flips
        should take place to avoid extreme bends in the curves due to
        the gravitational magnitude being sampled up to the maximum
        allowable force to hit the upper limit of the y-axis interval,
        as [lower percentile, upper percentile]. The default behavior if
        the parameter isn't set is to use the 10th and 90th percentile.

    change_spacing : int
        The minimum space on the x-axis in full steps that is required
        between gravitational direction changes, with hiher values
        resulting in increased smoothness. The parameter has to be small
        enough that the provided 'n_measure' parameter divided by the
        the 'change_spacing' parameter is equal to or larger than the
        'direction_maximum' parameter, i.e. the number of measurements
        divided by the minimum x-axis spacing has to be >= the maximum
        number of direction changes so that all possibilities will fit.

    start_force : float
        The x-axis point before which no y-axis deviation with regard to
        the projectile's starting point should happen. This is useful
        if a function perturbation should only happen after a certain
        point, which can be specified by setting this parameter.

    flat_value : float
        The copy of the parameter start_force that is made to make some
        of the calculations easier, making it a code-related parameter.

    log_scale : bool
        The indicator whether the measurements on the x-axis should be
        on a logarithmic scale, while retaining the behavior of a code
        calculation for a linear scale. This means that the steps will
        be equally-spaced when displayed with a logarithmic x-axis.

    random_launch : bool
        The indicator whether no initial zero launch angle is necessary,
        i.e. projectiles will start at random angles sampled uniformly
        between -90 and 90 degrees for each curve separately.

    print_points : list or None
        The list of numbers of curves generated that mark a milestone
        in a 10%-spaced progress scheme. This applies when 10 or more
        curves are to be generated, otherwise the parameter is None.

    perc : int or None
        The percentage in 10% steps indicating how much progress in
        generating viable curves that satisfy the constraints has been
        realised. This applies when 10 or more curves are to be
        generated, otherwise the parameter is None.

    progress_update : int
        The number of viable curves that satisfy the constraints and
        have been realised. Initialized as zero, this value is only used
        if less than 10 curves are to be generated by the tool.

    done_curves : int
        The overall number of viable curves that satisfy the constraints
        and have been realized. This value is returned and passed again
        at each call of this function, keeping on overview of the total
        realized number to print the correct progress information.

    Returns:
    --------
    curves: list
        The generated curves in a list, with one list element per curve.
        Each list elemenet contains two rows, the first for the x-axis
        measurement points and the second for the y-axis measurements.

    print_points : list or None
        The list of numbers of curves generated that mark a milestone
        in a 10%-spaced progress scheme. This applies when 10 or more
        curves are to be generated, otherwise the parameter is None.

    perc : int or None
        The percentage in 10% steps indicating how much progress in
        generating viable curves that satisfy the constraints has been
        realised. This applies when 10 or more curves are to be
        generated, otherwise the parameter is None.

    progress_update : int
        The number of viable curves that satisfy the constraints and
        have been realised. Initialized as zero, this value is only used
        if less than 10 curves are to be generated by the tool.

    done_curves : int
        The overall number of viable curves that satisfy the constraints
        and have been realized. This value is returned and passed again
        at each call of this function, keeping on overview of the total
        realized number to print the correct progress information.

    Attributes:
    -----------
    None
    """
    # Initialize an empty list for storing the curves later
    curves = []
    # Loop over the required total number of separate curves
    for curve in range(0, n_curves):
        # If no convergence point is given sample a random one
        if convergence_flag == True:
            y_convergence = uniform(y_interval[0], y_interval[1])
            convergence_point = [x_interval[0], y_convergence]
            if log_scale == True:
                convergence_point[0] = np.log10(convergence_point[0])
        elif log_scale == True:
             convergence_point[0] = np.log10(convergence_point[0])
        # Reset the start force if a flat state is requested
        if flat_state == True:
            start_force = flat_value
        # Generate the random force direction change points
        allowed = range(0, np.random.randint(0, direction_maximum + 1))
        if start_force == None:
            lower_range = int(np.multiply(len(steps), change_range[0]))
        else:
            diff_a = start_force - x_interval[0]
            diff_b = x_interval[1] - x_interval[0]
            diff_ratio = np.divide(diff_a, diff_b)
            flat_change = int(np.multiply(len(steps), diff_ratio))
            lower_standard = int(np.multiply(len(steps), change_range[0]))
            lower_range = np.maximum(lower_standard, flat_change)
        higher_range = int(np.multiply(len(steps), change_range[1]))
        sample_range = range(lower_range, higher_range)
        sample_number = np.random.randint(0, direction_maximum + 1)
        # Sample change points with the defined minimum space between
        change_points = []
        # Add the flat-start change point to the beginning
        if start_force != None:
            # Adapt the change points to allow the flat start
            change_points.append([flat_change])
        valid_counter = 0
        while valid_counter < sample_number:
            change_sample = sample(sample_range, 1)
            if np.asarray([np.abs(np.diff((change_points[i][0],
                                           change_sample[0])))[0]
                           >= change_spacing
                           for i in range(0, len(change_points))]).all():
                change_points.append(change_sample)
                valid_counter = valid_counter + 1
        change_points = np.sort(np.asarray(change_points).flatten())
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
        # Randomly sample the force depending on the maximum
        force = uniform(0, force_max)
        # Set the convergence point as the first start point
        start_point = convergence_point
        # Initialize a curve path with one point and a counter
        curve_path = [convergence_point]
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
                    if (change_ratio == None) or (part == 0):
                        force = uniform(0, force_max)
                    elif save_force == 0.0:
                        force = uniform(0, force_max)
                    else:
                        #limiter = np.minimum(save_force, force_max)
                        #force = uniform(0, np.multiply(limiter, change_ratio))

                        ratio_product = np.multiply(save_force, change_ratio)
                        limiter = np.minimum(force_max, ratio_product)
                        force = uniform(0, limiter)

                else:
                    force = 0.0
                    start_force = None
            # Save the force used to generate the partial curve
            save_force = force
            # Calculate the trajectory for the partial curve
            output = trajectory(force = force,
                                velocity = velocity,
                                direction = direction,
                                step_size = step_size,
                                start_point = start_point,
                                launch_angle = launch_angle,
                                partial_steps = partial_steps)
            # Assign the values from the trajectory function
            partial_path, last_point, launch_angle, velocity = output[0:4]
            # Update parameters for the next loop iteration
            start_point = last_point
            direction = -direction
            force = np.negative(force)


            # Get the maximum force to stay within the intervals
            rest_time = np.divide(x_interval[1] - last_point[0], velocity)
            max_range = y_interval[np.maximum(0, direction)] - last_point[1]
            abs_max = np.multiply(np.negative(direction), (max_range))
            spread = np.multiply(velocity, np.sin(launch_angle)) - abs_max
            force_max = np.divide(np.multiply(2, spread), np.square(rest_time))
            # Randomly sample the force depending on the maximum
            if change_ratio == None:
                        force = uniform(0, force_max)
            elif save_force == 0.0:
                force = uniform(0, force_max)
            else:
                limiter = np.minimum(save_force, force_max)
                force = uniform(0, np.multiply(limiter, change_ratio))


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
        # Delete curves that fall outside of the y-axis interval
        curves, delete_flag = deletion(curves = curves,
                                       y_interval = y_interval,
                                       n_curves = n_curves)
        # Print progress updates to inform about remaining time
        done_curves = done_curves + len(curves)
        if curve_request >= 10:
            if len(print_points) > 0:
                if len(curves) + done_curves >= print_points[0]:
                    perc = perc + 10
                    print("%d %%" % perc)
                    print_points = print_points[1:len(print_points)]
        elif (len(curves) > 0) and (delete_flag is False):
            progress_update = progress_update + 1
            print("%d curves generated" % progress_update)
    return curves, print_points, perc, progress_update, done_curves

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
    #log_interval = np.log10(x_interval)
    # Get the difference between starting and ending points
    #difference = np.diff([log_interval[0], log_interval[1]])
    # Compute the new step size over the logarithmic scale
    #step_size = np.divide(difference, n_measure - 1)
    # Calculate the steps with the new logarithmic step size
    #steps = [np.power(10, log_interval[0] + np.multiply(i, step_size))
    #             for i in range(0, n_measure)]

    # Get the logarithmic values for the start and end point
    log_interval = [np.log10(x_interval[0]), np.log10(x_interval[1])]
    # Change x-axis measurement points to a logarithmic scale
    steps = np.logspace(log_interval[0], log_interval[1], n_measure)
    # Return the steps and step size as the function output
    return steps

def trajectory(force,
               velocity,
               direction,
               step_size,
               start_point,
               launch_angle,
               partial_steps):
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
    # Initialize the horizontal displacement of the particle
    horizontal_displacement = [0.0]
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

def check(n_curves,
          x_interval,
          y_interval,
          n_measure,
          direction_maximum,
          convergence_point,
          log_scale,
          random_launch,
          right_convergence,
          change_range,
          change_spacing,
          change_ratio,
          start_force):
    """
    Check the user-provided parameter to make sure they are valid inputs.

    This function checks the parameters provided to the primary function
    to avoid any mishaps due to invalid inputs. If one or more parameters
    don't fulfill the requirements of the code, for example due to a wrong
    format, an error notification with an explanation for each invalid
    input parameter is printed, followed by the code's termination.

    Parameters:
    -----------
    n_curves : int
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

    n_measure : int
        The number of equally-spaced measurement points on the x-axis
        for each curve. If the parameter 'log_scale' is set to True, the
        points will be equally-spaced only if depicted on a logarithmic
        x-axis. Otherwise, they will be equally-spaced on linear scales.

    direction_maximum : int
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

    log_scale : bool
        The indicator whether the measurements on the x-axis should be
        on a logarithmic scale, while retaining the behavior of a code
        calculation for a linear scale. This means that the steps will
        be equally-spaced when displayed with a logarithmic x-axis.

    random_launch : bool
        The indicator whether no initial zero launch angle is necessary,
        i.e. projectiles will start at random angles sampled uniformly
        between -90 and 90 degrees for each curve separately.

    right_convergence : bool
        The indicator whether curves should converge on the right side
        instead of the left side. After computing the curves, their
        y-axis measurement vector will be flipped. If 'log_scale' is set
        to True and a value for 'start_force' is provided, this means
        that the 'start_force' threshold value for the first deviation
        from unity on the y-axis is calculated for left-side convergence
        before being flipped, which should be considered in the inputs.

    change_range : list
        The x-axis percentiles below and above which no gravity flips
        should take place to avoid extreme bends in the curves due to
        the gravitational magnitude being sampled up to the maximum
        allowable force to hit the upper limit of the y-axis interval,
        as [lower percentile, upper percentile]. The default behavior if
        the parameter isn't set is to use the 10th and 90th percentile.

    change_spacing : int
        The minimum space on the x-axis in full steps that is required
        between gravitational direction changes, with hiher values
        resulting in increased smoothness. The parameter has to be small
        enough that the provided 'n_measure' parameter divided by the
        the 'change_spacing' parameter is equal to or larger than the
        'direction_maximum' parameter, i.e. the number of measurements
        divided by the minimum x-axis spacing has to be >= the maximum
        number of direction changes so that all possibilities will fit.

    change_ratio : float
        The value by which the gravitational force of the previous
        partial trajectory of a given curve is multiplied to get the
        upper limit of the range from which the next partial trajectory
        of the same curve is sampled. Like 'change_spacing', this
        parameter is a way to enforce further smoothness.

    start_force : float
        The x-axis point before which no y-axis deviation with regard to
        the projectile's starting point should happen. This is useful
        if a function perturbation should only happen after a certain
        point, which can be specified by setting this parameter.

    Returns:
    --------
    None

    Attributes:
    -----------
    None
    """
    # Create a boolean vector to mark all incorrect inputs
    incorrect_inputs = np.zeros(15, dtype = bool)
    # Check if the number of curves is a positive integer
    if type(n_curves) is not int:
        incorrect_inputs[0] = True
    elif n_curves < 1:
        incorrect_inputs[0] = True
    # Check if the x-axis interval is a list of two floats
    if type(x_interval) is not list:
        incorrect_inputs[1] = True
    elif ((len(x_interval) is not 2)
          or (type(x_interval[0]) is not float)
          or (type(x_interval[1]) is not float)):
        incorrect_inputs[1] = True
    # Check if the y-axis interval is a list of two floats
    if type(y_interval) is not list:
        incorrect_inputs[2] = True
    elif ((len(y_interval) is not 2)
          or (type(y_interval[0]) is not float)
          or (type(y_interval[1]) is not float)):
        incorrect_inputs[2] = True
    # Check if the number of measurements is a valid integer
    if ((type(n_measure) is not int)
        or (n_measure < 0)):
        incorrect_inputs[3] = True
    # Check whether the change maximum is a valid integer
    if ((type(direction_maximum) is not int)
        or (direction_maximum < 0)):
        incorrect_inputs[4] = True
    # Check if the convergence point is None or valid
    if ((type(convergence_point) is not list)
        and (convergence_point is not None)):
        incorrect_inputs[5] = True
    elif type(convergence_point) is list:
        if ((len(convergence_point) is not 2)
            or (type(convergence_point[0]) is not float)
            or (type(convergence_point[1]) is not float)):
            incorrect_inputs[5] = True
        elif convergence_point[0] != x_interval[0]:
            incorrect_inputs[5] = True
    # Check if the log-scale indicator is a boolean
    if type(log_scale) is not bool:
        incorrect_inputs[6] = True
    # Check if the random launch indicator is a boolean
    if type(random_launch) is not bool:
        incorrect_inputs[7] = True
    # Check if the convergence indicator is a boolean
    if type(right_convergence) is not bool:
        incorrect_inputs[8] = True
    # Check if the change percentiles are valid inputs
    if ((type(change_range) is not list)
        and (change_range is not None)):
        incorrect_inputs[9] = True
    elif change_range is not None:
        if ((len(change_range) is not 2)
              or (type(change_range[0]) is not float)
              or (type(change_range[1]) is not float)):
            incorrect_inputs[9] = True
        elif ((change_range[0] < 0)
              or (change_range[0] > 1)
              or (change_range[1] < 0)
              or (change_range[1] > 1)):
            incorrect_inputs[9] = True
    # Check if the change spacing is a valid input
    if change_spacing is not None:
        if type(change_spacing) is not int:
            incorrect_inputs[10] = True
        elif change_spacing <= 0:
            incorrect_inputs[10] = True
        elif change_spacing > (np.divide(n_measure, direction_maximum)):
            incorrect_inputs[10] = True
    # Check if the change ratio is a valid float
    if change_ratio is not None:
        if type(change_ratio) is not float:
            incorrect_inputs[11] = True
        elif change_ratio <= 0:
            incorrect_inputs[11] = True
    # Check if the first deviation point is a valid float
    if start_force is not None:
        if ((type(start_force) is not float)
            or (start_force < x_interval[0])
            or (start_force > x_interval[1])):
            incorrect_inputs[12] = True
    # Check whether inputs are valid for a log-scale
    if (log_scale is True):
        if ((np.log10(x_interval[0]).is_integer() is False)
            or (np.log10(x_interval[1]).is_integer() is False)):
            incorrect_inputs[13] = True
        if convergence_point is not None:
            if np.log10(convergence_point[0]).is_integer() is False:
                incorrect_inputs[14] = True
    # Define error messages for each unsuitable parameter input
    errors = ['ERROR: n_curves: Must be an integer > 0',
              'ERROR: x_interval: Must be a list of length 2, ' +
              'with each element being a single float value ' +
              'and x_interval[0] < x_interval[1]',
              'ERROR: y_interval: Must be a list of length 2, ' +
              'with each element being a single float value ' +
              'and y_interval[0] < y_interval[1]',
              'ERROR: n_measure: Must be an integer > 0',
              'ERROR: direction_maximum: Must be an integer > 0',
              'ERROR: convergence_point: Must be either None ' +
              'or a list of length 2, with each element being ' +
              'a single float value and the first element being ' +
              'identical to the first element of x_interval',
              'ERROR: log_scale: Must be a boolean value',
              'ERROR: random_launch: Must be a boolean value ',
              'ERROR: right_convergence: Must be a boolean value ',
              'ERROR: change_range: Must be either None or a ' +
              'list of length two, with each element being a ' +
              'single float value between 0.0 and 1.0',
              'ERROR: change_spacing: Must be either None or a ' +
              'single integer > 0, so that n_measures divived' +
              'by direction_maximum is equal to or larger than' +
              'the value provided for this parameter',
              'ERROR: change_ratio: Must be either None or a ' +
              'single float > 0',
              'ERROR: start_force: Must be either None or a ' +
              'float value between the first and the second ' +
              'element of x_interval',
              'ERROR: x_interval, log_scale: If log_scale is ' +
              'True, the float values in x_interval have to be ' +
              'valid log-scale values, e.g. 0.01 or 10.0',
              'ERROR: convergence_point, log_scale: If log_scale ' +
              'is True, the first element of convergence_points ' +
              'has to be a valid log-scale value, e.g. 0.01 or 10.0']
    # If there are unsuitable inputs, print errors and terminate
    if any(value == True for value in incorrect_inputs):
        for i in range(0, len(errors)):
            if incorrect_inputs[i] == True:
                print(errors[i])
        sys.exit()
