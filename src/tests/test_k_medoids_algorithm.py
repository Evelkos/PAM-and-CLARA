import numpy as np

from clustering_algorithms import KMedoidsAlgorithm, Point


class TestKMedoidsAlgorithm:
    def setup(self):
        self.points = [
            Point(idx=1, coordinates=np.array([0, 1]), coordinates_names=["x", "y"]),
            Point(idx=2, coordinates=np.array([2, 3]), coordinates_names=["x", "y"]),
            Point(idx=3, coordinates=np.array([4, 5]), coordinates_names=["x", "y"]),
            Point(idx=4, coordinates=np.array([6, 7]), coordinates_names=["x", "y"]),
            Point(idx=5, coordinates=np.array([8, 9]), coordinates_names=["x", "y"]),
        ]

    def test_get_initial_medoids_indices(self):
        medoids = KMedoidsAlgorithm.get_initial_medoids_indices(self.points, 2, 12)
        assert len(medoids) == 2

        medoids = KMedoidsAlgorithm.get_initial_medoids_indices(self.points, 3, 12)
        assert len(medoids) == 3

    def test_prepare_medoids(self):
        algorithm = KMedoidsAlgorithm(self.points, clusters_num=2, labels=["x", "y"])

        # override medoids indices
        new_medoids_indices = [self.points[1].idx, self.points[3].idx]
        algorithm.medoids_indices = new_medoids_indices

        # prepare list of medoids
        medoids = algorithm.prepare_medoids()

        # correct medoids have been choosen
        assert medoids[0] == self.points[1]
        assert medoids[1] == self.points[3]
