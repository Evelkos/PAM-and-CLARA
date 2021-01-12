from clustering_algorithms import CLARA, PAM, get_initial_points
from data_loaders import load_data
from timer import Timer
from visualizers import plot_data

# FILENAME = "datasets/artificial/sizes3.arff"
FILENAME = "datasets/artificial/zelnik4.arff"
# FILENAME = "datasets/artificial/xclara.arff"


def run_clara(data, points):
    if not points:
        points = get_initial_points(data["df"], data["coordinates_columns"])
    clara = CLARA(points, len(data["classes"]), labels=data["classes"])
    clara.run()
    return clara.get_result_df()


def run_pam(data, points):
    if not points:
        points = get_initial_points(data["df"], data["coordinates_columns"])
    pam = PAM(points, len(classes))
    pam.run()
    return pam.get_result_df()


if __name__ == "__main__":
    data = load_data(FILENAME)
    # plot_data(data["df"], data["classes"], data["class_column"])

    # result = run_pam(data)

    points = get_initial_points(data["df"], data["coordinates_columns"])

    t = Timer()
    for iteration in range(10):
        t.start()
        result = run_clara(data, points)
        t.stop()
        print(t.time)

    # plot_data(
    #     result, data["classes"], "cluster", attributes_names=data["coordinates_columns"]
    # )
