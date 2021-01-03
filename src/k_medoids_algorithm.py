import random

import numpy as np

from point import Point


class KMedoidsAlgorithm:
    def __init__(self, df, clusters_num=2, labels=None, points = None, seed=44):
        random.seed(seed)
        self.seed = seed

        self.df = df
        self.clusters_num = clusters_num
        self.labels = labels

        self.points = points if points else self.get_initial_points(df)
        self.medoids_indices = self.get_initial_medoids_indices(seed)

    def get_initial_points(self, df):
        points = []
        for item in df.iterrows():
            idx, row = item
            coordinates = np.array([row["x"], row["y"]])
            point = Point(idx=idx, coordinates=coordinates)
            points.append(point)
        return points

    def get_initial_medoids_indices(self, seed=44):
        """
        Arbitrary selection of `cluster_num` objects.
        """
        points_indices = [point.idx for point in self.points]
        return random.sample(points_indices, self.clusters_num)


    def prepare_medoids(self):
        return [point for point in self.points if point.idx in self.medoids_indices]

    def update_clusters_assignment(self):
        medoids = self.prepare_medoids()
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

    def swap_medoids(self, old_medoid, new_medoid):
        self.medoids_indices.remove(old_medoid.idx)
        self.medoids_indices.append(new_medoid.idx)
