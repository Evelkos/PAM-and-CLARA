import random

import pandas as pd


class KMedoidsAlgorithm:
    def __init__(self, points, clusters_num=2, labels=None):
        self.points = points
        self.clusters_num = clusters_num
        self.labels = labels

        self.medoids_indices = self.get_initial_medoids_indices(
            self.points, clusters_num
        )

    @staticmethod
    def get_initial_medoids_indices(source_points, clusters_num, seed=44):
        """
        Arbitrary selection of `cluster_num` objects.
        """
        points_indices = [point.idx for point in source_points]
        return random.sample(points_indices, clusters_num)

    def prepare_medoids(self):
        return [point for point in self.points if point.idx in self.medoids_indices]

    def update_clusters_assignment(self):
        medoids = self.prepare_medoids()
        for point in self.points:
            point.update_cluster_assignment(medoids)

    def get_labels_mapper(self):
        if self.labels and len(self.labels) == self.clusters_num:
            return {
                medoid_idx: label
                for label, medoid_idx in zip(self.labels, self.medoids_indices)
            }
        return {medoid_idx: idx for idx, medoid_idx in enumerate(self.medoids_indices)}

    def get_result_df(self):
        rows = {column: [] for column in self.points[0].get_data().keys()}
        for point in self.points:
            for key, value in point.get_data().items():
                rows[key].append(value)

        result = pd.DataFrame(rows)
        result["cluster"] = result["nearest_medoid"]
        result["cluster"].replace(self.get_labels_mapper(), inplace=True)
        return result

    def swap_medoids(self, old_medoid, new_medoid):
        self.medoids_indices.remove(old_medoid.idx)
        self.medoids_indices.append(new_medoid.idx)
