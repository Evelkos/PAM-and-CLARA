import random

from clustering_algorithms import CLARA, PAM
from data_loaders import load_data
from visualizers import plot_data

FILENAME = "datasets/artificial/zelnik4.arff"
# FILENAME = "datasets/artificial/xclara.arff"


if __name__ == "__main__":

    df, classes = load_data(FILENAME)
    # plot_data(df, classes)
    # import pdb

    # pdb.set_trace()

    # pam = PAM(df, len(classes), seed=random.random())
    # pam.run(100)
    # result = pam.get_result_df()
    # plot_data(result, classes, "cluster")

    clara = CLARA(df, len(classes), seed=random.random())
    clara.run()
    result = clara.get_result_df()
    plot_data(result, classes, "cluster")
