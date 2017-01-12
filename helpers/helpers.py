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


