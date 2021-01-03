import random

import numpy as np

from point import Point
from k_medoids_algorithm import KMedoidsAlgorithm


class PAM(KMedoidsAlgorithm):
    def __init__(self, df, clusters_num=2, labels=None, points = None, seed=44):
        super().__init__(df=df, clusters_num=clusters_num, labels=labels, points=points, seed=seed)
        self.medoids = self.prepare_medoids()

    def compute_replacement_cost(self, old_medoid, new_medoid):
        cost = 0
        for point in self.points:
            if not point == new_medoid and point not in self.medoids:
                cost += point.compute_medoid_replacement_cost(
                    old_medoid, new_medoid, self.medoids
                )

        return cost

    def get_best_replacement_for_medoid(self, old_medoid):
        replacements = []
        for new_medoid in self.points:
            if new_medoid not in self.medoids:
                cost = self.compute_replacement_cost(old_medoid, new_medoid)
                replacements.append({"cost": cost, "new_medoid": new_medoid})

        best_replacement = min(replacements, key=lambda x: x["cost"])
        return best_replacement["new_medoid"], best_replacement["cost"]

    def run(self, max_iterations=None):
        iteration = 0
        while True:
            iteration += 1
            print(f"Iteration {iteration}")

            self.medoids = self.prepare_medoids()
            self.update_clusters_assignment()

            replacements = []
            for old_medoid in self.medoids:
                new_medoid, cost = self.get_best_replacement_for_medoid(old_medoid)
                replacements.append((cost, old_medoid, new_medoid))

            best_replacement = min(replacements, key=lambda x: x[0])
            self.swap_medoids(best_replacement[1], best_replacement[2])

            if best_replacement[0] >= 0:
                break

        self.update_clusters_assignment()
