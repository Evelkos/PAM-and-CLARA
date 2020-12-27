import matplotlib.pyplot as plt


def get_cmap(n, name="nipy_spectral"):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB.
    """
    return plt.cm.get_cmap(name, n)


def plot_data(data, classes, label="label"):
    # each class from `classes` needs to have different color.
    cmap = get_cmap(len(classes))
    colors = data[label].apply(cmap)

    plt.scatter(data["x"], data["y"], c=colors)
    plt.show()
