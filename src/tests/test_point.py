import numpy as np
import pandas as pd

from clustering_algorithms import Point, get_initial_points


def test_get_initial_points():
    # prepare dataframe to get initial points from
    df = pd.DataFrame(
        np.array(
            [
                [0.1, 1.2, "a"],
                [2.1, 3.2, "b"],
                [4.1, 5.2, "b"],
                [6.1, 7.2, "a"],
            ]
        ),
        columns=["x", "y", "label"],
    )
    points = get_initial_points(df)

    assert len(points) == 4

    # test labels and coordinates
    assert (points[0].coordinates == np.array([0.1, 1.2])).all()
    assert (points[1].coordinates == np.array([2.1, 3.2])).all()
    assert (points[2].coordinates == np.array([4.1, 5.2])).all()
    assert (points[3].coordinates == np.array([6.1, 7.2])).all()

    for point in points:
        point.nearest_medoid = None
        point.nearest_medoid_distance = None
        point.second_nearest_medoid = None
        point.second_nearest_medoid_distance = None


class TestPoint:
    def test_get_data(self):
        point = Point(idx=1, coordinates=[66, 99], coordinates_names=["x1", "x2"])
        # override some of point's attributes to test function properly
        point.nearest_medoid = Point(2, [2, 2], ["x1", "x2"])
        point.second_nearest_medoid = Point(3, [3, 3], ["x1", "x2"])

        point.nearest_medoid_distance = 34
        point.second_nearest_medoid_distance = 78

        data = point.get_data()

        assert data["idx"] == 1
        assert data["nearest_medoid"] == 2
        assert data["nearest_medoid_distance"] == 34
        assert data["second_nearest_medoid"] == 3
        assert data["second_nearest_medoid_distance"] == 78
        assert data["x1"] == 66
        assert data["x2"] == 99

    def test_update_cluster_assignment_when_point_is_not_a_medoid(self):
        point = Point(idx=999, coordinates=0, coordinates_names="x")
        medoids = [
            Point(0, np.array([12]), "x"),
            Point(1, np.array([11]), "x"),
            Point(2, np.array([10]), "x"),
            Point(3, np.array([11]), "x"),
            Point(4, np.array([12]), "x"),
        ]
        point.update_cluster_assignment(medoids)

        assert point.nearest_medoid == medoids[2]
        assert point.nearest_medoid_distance == 10
        assert point.second_nearest_medoid == medoids[1]
        assert point.second_nearest_medoid_distance == 11

    def test_update_cluster_assignment_when_point_is_a_medoid(self):
        medoids = [
            Point(0, np.array([12]), "x"),
            Point(1, np.array([11]), "x"),
            Point(2, np.array([10]), "x"),
            Point(3, np.array([11]), "x"),
            Point(4, np.array([12]), "x"),
        ]
        point = medoids[3]
        point.update_cluster_assignment(medoids)

        assert point.nearest_medoid == medoids[3]
        assert point.nearest_medoid_distance == 0
        assert point.second_nearest_medoid is None
        assert point.second_nearest_medoid_distance is None

    def test_compute_medoid_replacement_cost_when_point_is_a_medoid(self):
        points = [
            Point(0, np.array([10]), "x"),
            Point(1, np.array([11]), "x"),
            Point(2, np.array([12]), "x"),
            Point(3, np.array([13]), "x"),
            Point(4, np.array([14]), "x"),
        ]
        medoids = [points[0], points[4]]
        old_medoid = medoids[0]
        new_medoid = medoids[1]
        assert (
            new_medoid.compute_medoid_replacement_cost(old_medoid, new_medoid, medoids)
            == 0
        )

    def test_cost_of_replacing_nearest_medoid_with_more_similar_medoid(self):
        # create medoids
        nearest_medoid = Point(1, np.array([10]), "x")
        second_nearest_medoid = Point(2, np.array([20]), "x")
        medoids = [nearest_medoid, second_nearest_medoid]

        # create point and assign medoids
        point = Point(0, np.array([0]), "x")
        point.nearest_medoid = nearest_medoid
        point.nearest_medoid_distance = 10
        point.second_nearest_medoid = second_nearest_medoid
        point.second_nearest_medoid_distance = 20

        new_medoid = Point(3, np.array([5]), "x")
        cost = point.compute_medoid_replacement_cost(
            nearest_medoid, new_medoid, medoids
        )
        assert cost == new_medoid.coordinates[0] - nearest_medoid.coordinates[0]

    def test_cost_of_replacing_nearest_medoid_with_less_similar_medoid(self):
        # create medoids
        nearest_medoid = Point(1, np.array([10]), "x")
        second_nearest_medoid = Point(2, np.array([20]), "x")
        medoids = [nearest_medoid, second_nearest_medoid]

        # create point and assign medoids
        point = Point(0, np.array([0]), "x")
        point.nearest_medoid = nearest_medoid
        point.nearest_medoid_distance = 10
        point.second_nearest_medoid = second_nearest_medoid
        point.second_nearest_medoid_distance = 20

        new_medoid = Point(3, np.array([25]), "x")
        cost = point.compute_medoid_replacement_cost(
            nearest_medoid, new_medoid, medoids
        )
        assert (
            cost == second_nearest_medoid.coordinates[0] - nearest_medoid.coordinates[0]
        )

    def test_cost_of_replacing_some_medoid_with_points_new_nearest_medoid(self):
        # create medoids
        nearest_medoid = Point(1, np.array([10]), "x")
        second_nearest_medoid = Point(2, np.array([20]), "x")
        medoid = Point(2, np.array([40]), "x")
        medoids = [nearest_medoid, second_nearest_medoid, medoid]

        # create point and assign medoids
        point = Point(0, np.array([0]), "x")
        point.nearest_medoid = nearest_medoid
        point.nearest_medoid_distance = 10
        point.second_nearest_medoid = second_nearest_medoid
        point.second_nearest_medoid_distance = 20

        new_medoid = Point(3, np.array([5]), "x")
        cost = point.compute_medoid_replacement_cost(medoid, new_medoid, medoids)
        assert cost == new_medoid.coordinates[0] - nearest_medoid.coordinates[0]

    def test_cost_of_replacing_medoid_with_distant_medoid(self):
        # create medoids
        nearest_medoid = Point(1, np.array([10]), "x")
        second_nearest_medoid = Point(2, np.array([20]), "x")
        medoid = Point(2, np.array([40]), "x")
        medoids = [nearest_medoid, second_nearest_medoid, medoid]

        # create point and assign medoids
        point = Point(0, np.array([0]), "x")
        point.nearest_medoid = nearest_medoid
        point.nearest_medoid_distance = 10
        point.second_nearest_medoid = second_nearest_medoid
        point.second_nearest_medoid_distance = 20

        new_medoid = Point(3, np.array([555]), "x")
        cost = point.compute_medoid_replacement_cost(medoid, new_medoid, medoids)
        assert cost == 0

    def test_compute_distance(self):
        point = Point(0, np.array([123]), "x")
        other = Point(1, np.array([9458]), "x")
        assert point.compute_distance(other) == 9458 - 123
