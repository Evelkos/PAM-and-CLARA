import random

from data_loader import load_data
from pam import PAM
from plotting import plot_data

# FILENAME = "datasets/artificial/zelnik4.arff"
FILENAME = "datasets/artificial/xclara.arff"


if __name__ == "__main__":

    df, classes = load_data(FILENAME)
    # plot_data(df, classes)

    pam = PAM(df, len(classes) - 1, seed=random.random())
    pam.run(100)
    result = pam.get_result_df()
    plot_data(result, classes, "cluster")
