import random

from data_loader import load_data
from pam import PAM
from k_medoids_algorithm import KMedoidsAlgorithm
from plotting import plot_data
from utils import compute_distance

# FILENAME = "datasets/artificial/zelnik4.arff"
FILENAME = "datasets/artificial/zelnik4.arff"
# FILENAME = "datasets/artificial/xclara.arff"


if __name__ == "__main__":

    df, classes = load_data(FILENAME)
    # plot_data(df, classes)

    pam = PAM(df, clusters_num=len(classes), seed=random.random())
    pam.run()
    result = pam.get_result_df()
    plot_data(result, classes, "cluster")

    # import pdb
    # pdb.set_trace()

    # pam.run(100)
    # result = pam.get_result_df()
    # plot_data(result, classes, "cluster")
    # algorithm = KMedoidsAlgorithm(df, ["x", "y"], len(classes))
    # algorithm.calculate_distance_to_medoids(compute_distance)
