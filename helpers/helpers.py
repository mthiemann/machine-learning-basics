import numpy as np
import matplotlib.pyplot as plt


def normalize_features(features):
    """Normalizes the passed array of features

    Args:
        features: An array containing the values for all features.
                    (expects a column for each feature and the
                        first column to be all ones)

    Returns:
        A tuple containing
        the normalized feature array,
        an array with the mean for each column,
        an array with the standard deviation for each column
    """

    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)

    for i in range(1, features.shape[1]):
        features[:, i] = (features[:, i] - mu[i]) / sigma[i]

    return features, mu, sigma


def map_features(feature_1, feature_2, degree=6):
    """Maps features to a specified degree and adds
        a column of ones as the first column

    Args:
        feature_1: A list containing all values for feature 1.
        feature_2: A list containing all values for feature 2.
        degree: The degree that is used for the mapping (default = 6)

    Returns:
        An array with the mapped features to the specified degree.
    """

    # determine number of columns of feature map
    n = int((2 + degree + 1) / 2 * (degree) + 1)
    m = len(feature_1)

    featureMap = np.zeros(shape=(m, n))  # create empty feature map
    featureMap[:, 0] = np.ones(m)  # add column of ones
    c = 1

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            # assign new feature column to feature map
            featureMap[:, c] = np.power(feature_1, i - j) \
                * np.power(feature_2, j)
            c += 1
    return featureMap


def label_plot(figure, x_axis, y_axis):
    plt.figure(figure)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)


def plot_fitted_line(train_x, theta, padding=0.02):
    """Plots a fitted line using the minimum and maximum
        values of train_x as starting points including
        the passed padding

    Args:
        train_x: A 2D array with one column
        theta: A 2x1 array
        padding: The padding for the extreme values
                    of train_x (default = 0.02)
    """

    minimum, maximum = get_min_max(train_x, padding)
    plot_x = (minimum, maximum)
    plot_y = (theta[0] + theta[1] * plot_x)

    plt.plot(plot_x, plot_y)
    plt.show()


def plot_unclassified_data(train_x, train_y, show=True,
                           figure_labels=["Figure 1"],
                           x_labels=["x-axis"], y_label="y-axis"):
    """Plots unclassified data for any number of features of train_x.
        A plot is created for each feature column.

    Args:
        train_x: A 2D array with one column for each feature
        train_y: A 2D array with one column
        show: Show the plot (default = true)
        figure_label: A list with figure labels
        x_label: A list with x-axis labels
        y_label: A list with y-axis labels
    """

    for i in range(train_x.shape[1]):
        label_plot(figure_labels[i], x_labels[i], y_label)

        train_x_column = train_x
        if train_x.shape[1] > 1:
            train_x_column = np.delete(train_x, i, axis=1)
        plt.plot(train_x_column, train_y, 'bo')

    if show:
        plt.show()


def plot_classified_data(train_X, train_Y, show=True,
                         figure_label="Figure 1",
                         x_label="x-axis", y_label="y-axis"):
    """Plots classified data for two features of train_x.

    Args:
        train_X: A 2D array with three columns (first column contains ones)
        train_Y: A 1D array
        show: Show the plot (default = true)
        **labels: A dictionary with the following keys
                    figure: Figure name
                    x_label: x-axis label
                    y_label: y-axis label
    """

    label_plot(figure_label, x_label, y_label)

    pos = np.where(train_Y == 1)[0]
    neg = np.where(train_Y == 0)[0]

    plt.plot(train_X[pos, 1], train_X[pos, 2], 'g+')
    plt.plot(train_X[neg, 1], train_X[neg, 2], 'ro')
    if show:
        plt.show()


def plot_decision_boundary(train_X, train_Y, theta, **labels):
    """ Plot decision boundarys for multiple dimensions

        Args:
            train_X: A numpy 2-D array (first column contains ones)
            train_Y: A numpy 1-D array
            theta: The corresponding weights for the features in train_X
            **labels: A dictionary with the following keys
                    figure: Figure name
                    x_label: x-axis label
                    y_label: y-axis label
    """
    plot_classified_data(train_X, train_Y, show=False, **labels)
    x1 = train_X[:, 1]
    x2 = train_X[:, 2]

    if(train_X.shape[1] <= 3):
        minimum, maximum = get_min_max(x1)

        plot_x = [minimum, maximum]
        # calculate y values for corresponding x's
        plot_y = -1 / theta[2] * (theta[1] * plot_x + theta[0])

        plt.ylim(get_min_max(x2))

        plt.plot(plot_x, plot_y)
        plt.show()
    else:
        u_bounds_min, u_bounds_max = get_min_max(x1)
        v_bounds_min, v_bounds_max = get_min_max(x2)

        u = np.linspace(u_bounds_min, u_bounds_max, 100)
        v = np.linspace(v_bounds_min, v_bounds_max, 100)
        z = np.zeros((u.size, v.size))
        d = get_degree(train_X.shape[1])

        for i in range(u.size):
            for j in range(v.size):
                # calculate hypothesis value for specified values
                z[i, j] = np.matmul(map_features(
                    np.array([u[i]]), np.array([v[j]]), degree=d), theta)

        np.transpose(z)
        plt.contour(u, v, z, levels=[0])
        plt.show()


def get_min_max(values, padding=0.02):
    """Get the minimum and maximum value of the list
        with respect to the supplied padding.

        The values are calculated as follows:
            minimum: (minimum value) - padding * range
            maximum: (maximum value) + padding * range

        Args:
            values: A list containing the values.
            padding: The padding for both values.

        Returns:
            A tuple with the minimum and maximum value of the list
                with respect to the supplied padding
    """
    minimum, maximum = np.amin(values), np.amax(values)
    v_range = maximum - minimum
    return (minimum - padding * v_range, maximum + padding * v_range)


def get_degree(num_values, has_bias_value=True):
    """Get the degree of a used function given the number of values.

    Args:
        num_vals: The number of values as an integer.
        has_bias_value: Set to true, if the function has
                        a one for the bias value (default = true).

    Returns:
        The degree of the function as an integer.
    """
    x = 0.25 if has_bias_value else 2.25
    return int(-1.5 + np.sqrt(x + 2 * num_values))


def get_accuracy(predictions, actual_values):
    """Get the accuracy of the model prediction in percent.

    Args:
        predictions: A list with the predicted values.
        real_values: A list with the actual values.

    Returns:
        A percent value which represents the accuracy of the model prediction.
    """
    return (1 - (np.count_nonzero(predictions - actual_values) /
            predictions.size)) * 100
