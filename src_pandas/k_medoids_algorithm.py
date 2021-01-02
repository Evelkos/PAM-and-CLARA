import numpy as np
import random


class KMedoidsAlgorithm:
    def __init__(self, df, coordinates_columns=["x", "y"], clusters_num=2, labels=None, medoids = [], seed=44):
        """
        Arguments:
            df: dataset
            coordinates_columns: list of columns with coordinates
            clusters_num: number of clusters
            seed: seed
        """
        random.seed(seed)
        self.seed = seed

        # dataset and new columns
        self.df = df
        self.coordinates_columns = coordinates_columns

        self.df["nearest_medoid"] = np.nan
        self.df["nearest_medoid_distance"] = np.nan
        self.df["second_nearest_medoid"] = np.nan
        self.df["second_nearest_medoid_distance"] = np.nan

        # tmp column used in calculations
        self.df["distance"] = np.nan

        self.clusters_num = clusters_num
        self.labels = labels

        # rows selected as medoids
        self.medoids_indices = medoids

    def get_labels_mapper(self):
        if self.labels:
            return {
                medoid_idx: label
                for label, medoid_idx in zip(self.labels, self.medoids_indices)
            }
        return {medoid_idx: idx for idx, medoid_idx in enumerate(self.medoids_indices)}

    def get_result_df(self):
        self.df.rename(columns={"nearest_medoid": "cluster"}, inplace=True)
        self.df.replace({"cluster": self.get_labels_mapper()}, inplace=True)
        return self.df

    def swap_medoids(self, old_medoid_idx, new_medoid_idx):
        self.medoids_indices.remove(old_medoid_idx)
        self.medoids_indices.append(new_medoid_idx)

    def get_coordinates(self, row):
        return row[self.coordinates_columns]

    def calculate_distance_to_medoid(self, medoid, compute_distance):
        medoid_coordinates = self.get_coordinates(medoid)
        return self.df.apply(lambda row: compute_distance(medoid_coordinates, self.get_coordinates(row)), axis=1)

    def update_points_assignment(self, compute_distance):
        for idx, medoid in self.df.iloc[self.medoids_indices].iterrows():
            distances = self.calculate_distance_to_medoid(medoid, compute_distance)

            # override second nearest medoid
            rows = self.df["second_nearest_medoid_distance"].isna() | (self.df["second_nearest_medoid_distance"].lt(distances) & self.df["nearest_medoid_distance"].gt(distances))
            self.df.loc[rows, "second_nearest_medoid"] = idx
            self.df.loc[rows, "second_nearest_medoid_distance"] = distances[rows]

            # override nearest_medoids
            rows = self.df["nearest_medoid_distance"].isna() | self.df["nearest_medoid_distance"].gt(distances)
            # for selected rows, current nearest medoids become second nearest medoids
            self.df.loc[rows, "second_nearest_medoid"] = self.df.loc[rows, "nearest_medoid"]
            self.df.loc[rows, "second_nearest_medoid_distance"] = self.df.loc[rows, "nearest_medoid_distance"]
            # for selected rows, analysed medoid becomes nearest nearest medoid
            self.df.loc[rows, "nearest_medoid"] = idx
            self.df.loc[rows, "nearest_medoid_distance"] = distances[rows]

        self.df["nearest_medoid"] = self.df["nearest_medoid"].astype(int)
        self.df["second_nearest_medoid"] = self.df["second_nearest_medoid"].astype(int)

    def compute_medoid_replacement_cost(self, old_medoid_idx, new_medoid_idx, compute_distance):
        """
        Compute partial cost of replacing medoid `medoid_to_change` by object
        `new_medoid`.

        Arguments:
            old_medoid_idx: index of the medoid that we want to replace by new object
            new_medoid_idx: index of the new medoid

        """
        # distance to the new medoid
        if new_medoid_idx in self.medoids_indices:
            return 0

        old_medoid = self.df.loc[old_medoid_idx]
        new_medoid = self.df.loc[new_medoid_idx]

        new_medoid_distance = self.calculate_distance_to_medoid(new_medoid, compute_distance)

        cost = 0

        # we want to replace medoid that represents points' cluster
        # second_nearest_medoid will be points' nearest medoid
        rows = self.df["nearest_medoid"].eq(old_medoid_idx) & self.df["second_nearest_medoid_distance"].le(new_medoid_distance)
        cost += sum(self.df.loc[rows, "second_nearest_medoid_distance"] - self.df.loc[rows, "nearest_medoid_distance"])
        # new medoid will be points' nearest medoid
        rows = self.df["nearest_medoid"].eq(old_medoid_idx) & self.df["second_nearest_medoid_distance"].gt(new_medoid_distance)
        cost = sum(new_medoid_distance[rows] - self.df.loc[rows, "nearest_medoid_distance"])

        # we want to replace medoid that does not represent points' cluster
        # # points' cluster will be the same as before
        # rows = self.df["nearest_medoid"].ne(old_medoid_idx) & self.df["nearest_medoid_distance"].le(new_medoid_distance)
        # cost += 0

        # new medoid will be points` nearest medoid
        rows = self.df["nearest_medoid"].ne(old_medoid_idx) & self.df["nearest_medoid_distance"].gt(new_medoid_distance)
        cost += sum(new_medoid_distance[rows] - self.df.loc[rows, "nearest_medoid_distance"])

        return cost


    # def calculate_distances(self, compute_distance):
    #     for idx, point in self.df.iterrows():
    #         # calculate distance from selected point to every other point
    #         point_coordinates = self.get_coordinates(point)
    #         self.df[f""] = self.df.apply(lambda row: compute_distance(point_coordinates, self.get_coordinates(row)), axis=1)


    #         # update nearest_medoid and nearest_medoid_distance
    #         # first assignment or better cluster found
    #         to_update = self.df['nearest_medoid_distance'] == np.nan or self.df['nearest_medoid_distance'] > self.df['distance']

    #         self.df.loc[to_update]['second_nearest_medoid'] = self.df.loc[to_update]['nearest_medoid']
    #         self.df.loc[to_update]['second_nearest_medoid_distance'] = self.df.loc[to_update]['nearest_medoid_distance']

    #         if self.nearest_medoid is np.nan or distance < self.nearest_medoid_distance:
    #             # update values connected to the second nearest medoid
    #             self.second_nearest_medoid = self.nearest_medoid
    #             self.second_nearest_medoid_distance = self.nearest_medoid_distance
    #             # update values connected to the nearest medoid
    #             self.nearest_medoid = medoid
    #             self.nearest_medoid_distance = distance

    #         # first assignment or better cluster found
    #         elif (
    #             self.second_nearest_medoid is np.nan
    #             or distance < self.second_nearest_medoid_distance
    #         ):
    #             self.second_nearest_medoid = medoid
    #             self.second_nearest_medoid_distance = distance


    # def determine_nearest_medoids(self):
    #   pass

    # def update_cluster_assignment(self, medoids):
    #     """
    #     Pick nearest and second nearest medois from given medoids list.
    #     Update Point's assignment.
    #     """
    #     if self in medoids:
    #         self.nearest_medoid = self
    #         return

    #     self.nearest_medoid = np.nan
    #     self.nearest_medoid_distance = np.nan
    #     self.second_nearest_medoid = np.nan
    #     self.second_nearest_medoid_distance = np.nan

    #     for medoid in medoids:
    #         distance = compute_distance(self.coordinates, medoid.coordinates)

    #         # first assignment or better cluster found
    #         if self.nearest_medoid is np.nan or distance < self.nearest_medoid_distance:
    #             # update values connected to the second nearest medoid
    #             self.second_nearest_medoid = self.nearest_medoid
    #             self.second_nearest_medoid_distance = self.nearest_medoid_distance
    #             # update values connected to the nearest medoid
    #             self.nearest_medoid = medoid
    #             self.nearest_medoid_distance = distance

    #         # first assignment or better cluster found
    #         elif (
    #             self.second_nearest_medoid is np.nan
    #             or distance < self.second_nearest_medoid_distance
    #         ):
    #             self.second_nearest_medoid = medoid
    #             self.second_nearest_medoid_distance = distance