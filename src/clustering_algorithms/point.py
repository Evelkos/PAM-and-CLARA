from typing import List

import numpy as np
import pandas as pd

from numpy.linalg import norm


def get_initial_points(df: pd.DataFrame, coordinates_names=["x", "y"]):
    points = []
    for item in df.iterrows():
        idx, row = item
        coordinates = np.array([row[name] for name in coordinates_names])
        point = Point(
            idx=idx, coordinates=coordinates, coordinates_names=coordinates_names
        )
        points.append(point)
    return points


class Point:
    def __init__(
        self, idx: int, coordinates: List[float], coordinates_names: List[str]
    ):
        if not len(coordinates) == len(coordinates_names):
            raise ValueError(
                "List of coordinates and list of coordinates's names "
                "need to be the same length"
            )

        self.idx = idx
        self.coordinates = coordinates
        self.coordinates_names = coordinates_names

        # Point's cluster (nearest medoid)
        self.nearest_medoid = None
        self.nearest_medoid_distance = None

        # Point's second nearest medoid
        self.second_nearest_medoid = None
        self.second_nearest_medoid_distance = None

    def compute_distance(self, other_point: "Point") -> float:
        """
        Compute distance between two points.

        Arguments:
            other_point: point

        Return:
            Linear distance between two points.

        """
        return norm(self.coordinates - other_point.coordinates)

    def get_data(self) -> dict:
        """
        Get info about Point.

        Returns:
            Dictionary with data about point:
                * idx
                * nearest_medoid
                * nearest_medoid_distance
                * second_nearest_medoid
                * second_nearest_medoid_distance
                * all coordinates with their names as keys

        """
        data = {
            "idx": self.idx,
            "nearest_medoid": self.nearest_medoid.idx,
            "nearest_medoid_distance": self.nearest_medoid_distance,
            "second_nearest_medoid": self.second_nearest_medoid.idx,
            "second_nearest_medoid_distance": self.second_nearest_medoid_distance,
        }
        data.update(
            {
                name: value
                for name, value in zip(self.coordinates_names, self.coordinates)
            }
        )
        return data

    def update_cluster_assignment(self, medoids: List["Point"]) -> None:
        """
        Choose nearest medoid and second nearest medoid from given `medoids` and assign
        them to point's nearest_medoid and second_nearest_medoid attributes. If point
        is a medoid, nearest_medoid attribute will be pointing to itself. Update
        point's nearest_medoid_distance and second_nearest_medoid_distance.

        Arguments:
            medoids: list of available medoids

        """
        self.nearest_medoid = np.nan
        self.nearest_medoid_distance = np.nan
        self.second_nearest_medoid = np.nan
        self.second_nearest_medoid_distance = np.nan

        for medoid in medoids:
            distance = self.compute_distance(medoid)

            # first assignment or better cluster found
            if self.nearest_medoid is np.nan or distance < self.nearest_medoid_distance:
                # update values connected to the second nearest medoid
                self.second_nearest_medoid = self.nearest_medoid
                self.second_nearest_medoid_distance = self.nearest_medoid_distance
                # update values connected to the nearest medoid
                self.nearest_medoid = medoid
                self.nearest_medoid_distance = distance

            # first assignment or better cluster found
            elif (
                self.second_nearest_medoid is np.nan
                or distance < self.second_nearest_medoid_distance
            ):
                self.second_nearest_medoid = medoid
                self.second_nearest_medoid_distance = distance

    def compute_medoid_replacement_cost(
        self, old_medoid: "Point", new_medoid: "Point", medoids: List["Point"]
    ) -> float:
        """
        Compute partial cost of replacing `old_medoid` by `new_medoid`.

        Arguments:
            old_medoid: medoid that we want to replace
            new_medoid: point that could replace old medoid
            medoids: list of currently available medoids

        Return:
            Cost of replacing `old_medoid` with `new_medoid`.

        """
        # medoid's cluster will not change, so cost is equal to 0
        if self in medoids and self is not old_medoid:
            return 0

        # distance to the new medoid
        new_medoid_distance = self.compute_distance(new_medoid)

        # we want to replace medoid that represents Point's cluster
        if self.nearest_medoid == old_medoid:
            if new_medoid_distance >= self.second_nearest_medoid_distance:
                # second_nearest_medoid will be Point's nearest_medoid
                return (
                    self.second_nearest_medoid_distance - self.nearest_medoid_distance
                )
            else:
                # new_medoid will be Point's nearest_medoid
                return new_medoid_distance - self.nearest_medoid_distance
        # we want to replace medoid that does not represent Point's cluster
        else:
            if new_medoid_distance >= self.nearest_medoid_distance:
                # Point's cluster will be the same as before
                return 0
            else:
                # new_medoid will be Point's nearest_medoid
                return new_medoid_distance - self.nearest_medoid_distance
