import matplotlib.pyplot as plt


def get_cmap(n, name="nipy_spectral"):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB.
    """
    return plt.cm.get_cmap(name, n)


def plot_data(data, classes, label_col="label", attributes_names=["x", "y"]):
    # each class from `classes` needs to have different color.
    color_map = get_cmap(len(classes))
    class_map = {label: idx for idx, label in enumerate(classes)}
    colors = data[label_col].apply(lambda x: color_map(class_map[x]))

    plt.scatter(data[attributes_names[0]], data[attributes_names[1]], c=colors)
    plt.show()
