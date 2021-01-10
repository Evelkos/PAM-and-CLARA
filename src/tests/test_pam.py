import numpy as np

from clustering_algorithms import PAM, Point


class TestPam:
    def test_get_best_replacement_for_medoid(self):
        points = [
            Point(idx=10, coordinates=np.array([0]), coordinates_names=["x"]),
            Point(idx=20, coordinates=np.array([1]), coordinates_names=["x"]),
            Point(idx=30, coordinates=np.array([2]), coordinates_names=["x"]),
            Point(idx=40, coordinates=np.array([100]), coordinates_names=["x"]),
            Point(idx=50, coordinates=np.array([101]), coordinates_names=["x"]),
            Point(idx=60, coordinates=np.array([102]), coordinates_names=["x"]),
        ]
        pam = PAM(points, clusters_num=2, labels=None)

        # choose medoids and update points assignment
        pam.medoids = [points[1], points[2]]
        pam.medoids_indices = [points[1].idx, points[2].idx]
        pam.update_clusters_assignment()

        # if we swap these points, we will get the best possible clustering
        old_medoid = points[2]
        new_medoid = points[4]

        best_replacement, cost = pam.get_best_replacement_for_medoid(old_medoid)
        assert best_replacement == new_medoid

    def test_get_best_replacement_for_medoid(self):
        points = [
            Point(idx=10, coordinates=np.array([0]), coordinates_names=["x"]),
            Point(idx=20, coordinates=np.array([1]), coordinates_names=["x"]),
            Point(idx=30, coordinates=np.array([2]), coordinates_names=["x"]),
            Point(idx=40, coordinates=np.array([100]), coordinates_names=["x"]),
            Point(idx=50, coordinates=np.array([101]), coordinates_names=["x"]),
            Point(idx=60, coordinates=np.array([102]), coordinates_names=["x"]),
        ]
        pam = PAM(points, clusters_num=2, labels=None)

        # choose medoids and update points assignment
        pam.medoids = [points[1], points[2]]
        pam.medoids_indices = [points[1].idx, points[2].idx]
        pam.update_clusters_assignment()

        # choose medoid that we want to replace
        old_medoid = points[2]

        assert pam.compute_replacement_cost(old_medoid, points[0]) == 3.0
        assert pam.compute_replacement_cost(old_medoid, points[1]) == 3.0
        assert pam.compute_replacement_cost(old_medoid, points[2]) == 0.0
        assert pam.compute_replacement_cost(old_medoid, points[3]) == -196.0
        assert pam.compute_replacement_cost(old_medoid, points[4]) == -196.0
        assert pam.compute_replacement_cost(old_medoid, points[5]) == -194.0
