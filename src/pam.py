import random

import numpy as np

from point import Point


class PAM:
    def __init__(self, df, clusters_num=2, labels=None, seed=44):
        # set of objects ->  O
        self.df = df
        self.clusters_num = clusters_num
        self.labels = labels

        self.medoids_indices = self.get_initial_medoids_indices(seed)
        self.points = self.get_initial_points(df)

    def get_initial_medoids_indices(self, seed=44):
        """
        Arbitrary selection of `cluster_num` objects.
        """
        random.seed(seed)
        idx_range = range(len(self.df))
        return random.sample(idx_range, self.clusters_num)

    def get_initial_points(self, df):
        points = []
        for item in df.iterrows():
            idx, row = item
            coordinates = np.array([row["x"], row["y"]])
            point = Point(idx=idx, coordinates=coordinates)
            points.append(point)
        return points

    def update_clusters_assignment(self):
        medoids = [self.points[idx] for idx in self.medoids_indices]
        for point in self.points:
            point.update_cluster_assignment(medoids)

    def get_labels_mapper(self):
        if self.labels:
            return {
                medoid_idx: label
                for label, medoid_idx in zip(self.labels, self.medoids_indices)
            }
        return {medoid_idx: idx for idx, medoid_idx in enumerate(self.medoids_indices)}

    def get_result_df(self):
        labels_mapper = self.get_labels_mapper()

        result_labels = []
        for point in self.points:
            label = labels_mapper[point.nearest_medoid.idx]
            result_labels.append(label)

        self.df["cluster"] = result_labels
        return self.df

    def compute_replacement_cost(self, old_medoid, new_medoid):
        medoids = [self.points[idx] for idx in self.medoids_indices]

        cost = 0
        for point in self.points:
            if not point == new_medoid and point not in medoids:
                cost += point.compute_medoid_replacement_cost(
                    old_medoid, new_medoid, medoids
                )

        return cost

    def get_best_replacement_for_medoid(self, old_medoid):
        medoids = [self.points[idx] for idx in self.medoids_indices]

        replacements = []
        for new_medoid in self.points:
            if new_medoid not in medoids:
                cost = self.compute_replacement_cost(old_medoid, new_medoid)
                replacements.append({"cost": cost, "new_medoid": new_medoid})

        best_replacement = min(replacements, key=lambda x: x["cost"])
        return best_replacement["new_medoid"], best_replacement["cost"]

    def swap_medoids(self, old_medoid, new_medoid):
        self.medoids_indices.remove(old_medoid.idx)
        self.medoids_indices.append(new_medoid.idx)

    def run(self, max_iterations=None):
        iteration = 0
        while True:
            iteration += 1
            print(f"Iteration {iteration}")

            if max_iterations is not None and iteration >= max_iterations:
                break

            medoids = [self.points[idx] for idx in self.medoids_indices]
            self.update_clusters_assignment()

            replacements = []
            for old_medoid in medoids:
                new_medoid, cost = self.get_best_replacement_for_medoid(old_medoid)
                replacements.append(
                    {"cost": cost, "old_medoid": old_medoid, "new_medoid": new_medoid}
                )

            best_replacement = min(replacements, key=lambda x: x["cost"])
            self.swap_medoids(
                best_replacement["old_medoid"], best_replacement["new_medoid"]
            )

            if best_replacement["cost"] >= 0:
                break

        self.update_clusters_assignment()
