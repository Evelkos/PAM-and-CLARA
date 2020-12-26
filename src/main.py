from data_loader import load_data
from plotting import plot_data

FILENAME = "datasets/artificial/2d-10c.arff"


if __name__ == "__main__":
    df, classes = load_data(FILENAME)
    plot_data(df, classes)
