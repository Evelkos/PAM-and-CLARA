import numpy as np
import pytest

from clustering_algorithms import KMedoidsAlgorithm, Point


class TestKMedoidsAlgorithm:
    def test_list_of_labels_with_incorrect_length_passed_to_init(self):
        points = [
            Point(1, np.array([0, 1]), ["x", "y"]),
            Point(2, np.array([2, 3]), ["x", "y"]),
            Point(3, np.array([4, 5]), ["x", "y"]),
        ]

        with pytest.raises(ValueError):
            KMedoidsAlgorithm(points, 2, labels=["1", "2", "3"])

    def test_get_initial_medoids_indices(self):
        points = [
            Point(idx=1, coordinates=np.array([0, 1]), coordinates_names=["x", "y"]),
            Point(idx=2, coordinates=np.array([2, 3]), coordinates_names=["x", "y"]),
            Point(idx=3, coordinates=np.array([4, 5]), coordinates_names=["x", "y"]),
            Point(idx=4, coordinates=np.array([6, 7]), coordinates_names=["x", "y"]),
            Point(idx=5, coordinates=np.array([8, 9]), coordinates_names=["x", "y"]),
        ]

        medoids = KMedoidsAlgorithm.get_initial_medoids_indices(points, 2)
        assert len(medoids) == 2
        medoids = KMedoidsAlgorithm.get_initial_medoids_indices(points, 3)
        assert len(medoids) == 3

    def test_prepare_medoids(self):
        points = [
            Point(idx=1, coordinates=np.array([0, 1]), coordinates_names=["x", "y"]),
            Point(idx=2, coordinates=np.array([2, 3]), coordinates_names=["x", "y"]),
            Point(idx=3, coordinates=np.array([4, 5]), coordinates_names=["x", "y"]),
            Point(idx=4, coordinates=np.array([6, 7]), coordinates_names=["x", "y"]),
            Point(idx=5, coordinates=np.array([8, 9]), coordinates_names=["x", "y"]),
        ]
        algorithm = KMedoidsAlgorithm(points, clusters_num=2, labels=["x", "y"])

        # override medoids indices
        new_medoids_indices = [points[1].idx, points[3].idx]
        algorithm.medoids_indices = new_medoids_indices

        # prepare list of medoids
        medoids = algorithm.prepare_medoids()

        # correct medoids have been choosen
        assert medoids[0] == points[1]
        assert medoids[1] == points[3]

    def test_update_clusters_assignment(self):
        # create 3 groups of points
        points = [
            Point(idx=10, coordinates=np.array([0, 0]), coordinates_names=["x", "y"]),
            Point(idx=11, coordinates=np.array([0, 1]), coordinates_names=["x", "y"]),
            Point(idx=12, coordinates=np.array([1, 0]), coordinates_names=["x", "y"]),
            Point(idx=13, coordinates=np.array([1, 1]), coordinates_names=["x", "y"]),
            Point(idx=14, coordinates=np.array([10, 0]), coordinates_names=["x", "y"]),
            Point(idx=15, coordinates=np.array([10, 1]), coordinates_names=["x", "y"]),
            Point(idx=16, coordinates=np.array([11, 0]), coordinates_names=["x", "y"]),
            Point(idx=17, coordinates=np.array([11, 1]), coordinates_names=["x", "y"]),
            Point(idx=18, coordinates=np.array([100, 0]), coordinates_names=["x", "y"]),
            Point(idx=19, coordinates=np.array([100, 1]), coordinates_names=["x", "y"]),
            Point(idx=20, coordinates=np.array([101, 0]), coordinates_names=["x", "y"]),
            Point(idx=21, coordinates=np.array([101, 1]), coordinates_names=["x", "y"]),
        ]
        algorithm = KMedoidsAlgorithm(points, clusters_num=3, labels=["a", "b", "c"])
        algorithm.medoids_indices = [10, 14, 18]
        # override medoids_indices to test function properly

        algorithm.update_clusters_assignment()
        for point in points:
            if point.idx in [10, 11, 12, 13]:
                assert point.nearest_medoid == points[0]
            elif point.idx in [14, 15, 16, 17]:
                assert point.nearest_medoid == points[4]
            else:
                assert point.nearest_medoid == points[8]

    def test_labels_mapper_when_specified_labels(self):
        points = [
            Point(idx=10, coordinates=np.array([0, 0]), coordinates_names=["x", "y"]),
            Point(idx=11, coordinates=np.array([0, 1]), coordinates_names=["x", "y"]),
            Point(idx=12, coordinates=np.array([10, 0]), coordinates_names=["x", "y"]),
            Point(idx=13, coordinates=np.array([10, 1]), coordinates_names=["x", "y"]),
        ]
        algorithm = KMedoidsAlgorithm(points, 2, labels=["label_1", "label_2"])

        # choose medoids indices
        algorithm.medoids_indices = [13, 11]
        mapper = algorithm.get_labels_mapper()
        assert len(mapper) == 2
        assert mapper[13] == "label_1"
        assert mapper[11] == "label_2"

        # choose the same indices, but with different order
        algorithm.medoids_indices = [11, 13]
        mapper = algorithm.get_labels_mapper()
        assert len(mapper) == 2
        assert mapper[11] == "label_1"
        assert mapper[13] == "label_2"

    def test_labels_mapper_when_labels_are_not_specified(self):
        points = [
            Point(idx=10, coordinates=np.array([0, 0]), coordinates_names=["x", "y"]),
            Point(idx=11, coordinates=np.array([0, 1]), coordinates_names=["x", "y"]),
            Point(idx=12, coordinates=np.array([10, 0]), coordinates_names=["x", "y"]),
            Point(idx=13, coordinates=np.array([10, 1]), coordinates_names=["x", "y"]),
        ]
        algorithm = KMedoidsAlgorithm(points, 2, labels=None)

        # choose medoids indices
        algorithm.medoids_indices = [12, 13]
        mapper = algorithm.get_labels_mapper()
        assert len(mapper) == 2
        assert mapper[12] == 0
        assert mapper[13] == 1

    def test_result_df(self):
        point_1 = Point(
            idx=1, coordinates=np.array([0, 0]), coordinates_names=["x", "y"]
        )
        medoid_1 = Point(
            idx=11, coordinates=np.array([1, 0]), coordinates_names=["x", "y"]
        )

        point_2 = Point(
            idx=2, coordinates=np.array([101, 0]), coordinates_names=["x", "y"]
        )
        medoid_2 = Point(
            idx=22, coordinates=np.array([100, 0]), coordinates_names=["x", "y"]
        )

        point_1.nearest_medoid = medoid_1
        point_1.nearest_medoid_distance = 1
        point_1.second_nearest_medoid = medoid_2
        point_1.second_nearest_medoid_distance = 100

        point_2.nearest_medoid = medoid_2
        point_2.nearest_medoid_distance = 1
        point_2.second_nearest_medoid = medoid_1
        point_2.second_nearest_medoid_distance = 100

        medoid_1.nearest_medoid = medoid_1
        medoid_1.nearest_medoid_distance = 0
        medoid_1.second_nearest_medoid = medoid_2
        medoid_1.second_nearest_medoid_distance = 99

        medoid_2.nearest_medoid = medoid_2
        medoid_2.nearest_medoid_distance = 0
        medoid_2.second_nearest_medoid = medoid_1
        medoid_2.second_nearest_medoid_distance = 99

        points = [point_1, point_2, medoid_1, medoid_2]
        algorithm = KMedoidsAlgorithm(points, 2, labels=None)

        df = algorithm.get_result_df()
        points_mapper = {point.idx: point for point in points}

        assert len(df) == 4
        for _, row in df.iterrows():
            point = points_mapper[row.idx]
            assert row["nearest_medoid"] == point.nearest_medoid.idx
            assert row["x"] == point.coordinates[0]
            assert row["y"] == point.coordinates[1]

    def test_swap_medoids(self):
        points = [
            Point(idx=10, coordinates=np.array([0, 0]), coordinates_names=["x", "y"]),
            Point(idx=11, coordinates=np.array([0, 1]), coordinates_names=["x", "y"]),
            Point(idx=12, coordinates=np.array([10, 0]), coordinates_names=["x", "y"]),
            Point(idx=13, coordinates=np.array([10, 1]), coordinates_names=["x", "y"]),
        ]
        algorithm = KMedoidsAlgorithm(points, 2, labels=None)

        old_medoid = points[0]
        other_medoid = points[1]
        algorithm.medoids_indices = [old_medoid.idx, other_medoid.idx]

        new_medoid = points[2]
        algorithm.swap_medoids(old_medoid, new_medoid)

        assert len(algorithm.medoids_indices) == 2
        assert old_medoid.idx not in algorithm.medoids_indices
        assert other_medoid.idx in algorithm.medoids_indices
        assert new_medoid.idx in algorithm.medoids_indices
