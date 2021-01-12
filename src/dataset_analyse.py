import csv
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from scipy.io import arff

from clustering_algorithms import CLARA, PAM, get_initial_points
from data_loaders import load_data
from timer import Timer
from visualizers import plot_data

DIRECTORY = "datasets/artificial"

FILES = [
    "insect.arff",
    "flame.arff",
    "zelnik3.arff",
    "zelnik5.arff",
    "R15.arff",
    "square3.arff",
    "sizes5.arff",
    "cure-t1-2000n-2D.arff",
    "xclara.arff",
    "s-set1.arff",
]

if __name__ == "__main__":
    # files = [f for f in listdir(DIRECTORY) if isfile(join(DIRECTORY, f))]
    files = FILES
    for file in files:
        try:
            filepath = join(DIRECTORY, file)
            data = load_data(filepath)

            points = get_initial_points(data["df"], data["coordinates_columns"])

            clara_time = []
            pam_time = []
            t = Timer()
            for iteration in range(10):
                # count clara time
                t.start()
                clara = CLARA(points, len(data["classes"]), labels=data["classes"])
                clara.run()
                t.stop()
                clara_time.append(t.time)

                # # count pam time
                # t.start()
                # pam = PAM(points, len(data["classes"]), labels=data["classes"])
                # pam.run()
                # t.stop()
                # pam_time.append(t.time)

            results = {
                "filename": file,
                "classes": len(data["classes"]),
                "coordinates_attributes": len(data["coordinates_columns"]),
                "points": len(points),
                "mean_time_clara": np.mean(clara_time),
                # "mean_time_pam": np.mean(pam_time),
            }

            with open("results_clara.csv", "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=results.keys())
                writer.writerow(results)
        except:
            pass
