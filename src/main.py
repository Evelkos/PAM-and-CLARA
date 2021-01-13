from clustering_algorithms import CLARA, PAM, get_initial_points
from data_loaders import load_data
from timer import Timer
from visualizers import plot_data

# FILENAME = "datasets/artificial/sizes3.arff"
FILENAME = "datasets/artificial/zelnik4.arff"
# FILENAME = "datasets/artificial/xclara.arff"
# FILENAME = "datasets/real-world/glass.arff"


def run_clara(data, points):
    clara = CLARA(points, len(data["classes"]), labels=data["classes"])
    clara.run()
    return clara.get_result_df()


def run_pam(data, points):
    pam = PAM(points, len(data["classes"]), labels=data["classes"])
    pam.run()
    return pam.get_result_df()


if __name__ == "__main__":
    data = load_data(FILENAME)
    # plot_data(data["df"], data["classes"], data["class_column"])

    points = get_initial_points(data["df"], data["coordinates_columns"])
    # result = run_clara(data, points)
    result = run_pam(data, points)
    plot_data(
        result, data["classes"], "cluster", attributes_names=data["coordinates_columns"]
    )
