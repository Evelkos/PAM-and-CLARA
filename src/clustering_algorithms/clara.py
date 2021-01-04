import random
from statistics import mean
from typing import List

from clustering_algorithms.k_medoids_algorithm import KMedoidsAlgorithm
from clustering_algorithms.pam import PAM
from clustering_algorithms.point import Point


class CLARA(KMedoidsAlgorithm):
    def __init__(
        self,
        points: List[Point],
        clusters_num: int = 2,
        labels: List["str"] = None,
        samples_num: int = None,
    ):
        super().__init__(points=points, clusters_num=clusters_num, labels=labels)

        self.update_clusters_assignment()
        self.best_medoids = self.medoids_indices
        self.best_dissimilarity = self.calculate_dissimilarity()

        # number of samples that will be passed to the PAM algorithm
        if samples_num:
            self.samples_num = samples_num
        else:
            self.samples_num = min(40 + 2 * clusters_num, len(self.points))

    def draw_samples(self) -> List[Point]:
        """
        Draw a random sample of `samples_num` points from the entire dataset.

        Return:
            List of randomly selected points.

        """
        return random.sample(self.points, self.samples_num)

    def calculate_dissimilarity(self) -> float:
        """
        Calculate dissimilarity for Clara algorithm as a mean distance between
        points and their nearest medoids.

        Return:
            mean distance between points and medoids that represent their clusters

        """
        return mean(
            [point.compute_distance(point.nearest_medoid) for point in self.points]
        )

    def run(self) -> None:
        """
        Run CLARA algorithm. Use clara_instance.get_result_df() to fetch the results.

        """
        for idx in range(5):
            # call PAM to find medoids of the small sample
            pam = PAM(
                points=self.draw_samples(),
                clusters_num=self.clusters_num,
                labels=self.labels,
            )
            pam.run()

            # determine the most similar medoid for each point from dataset
            new_medoids = [medoid.idx for medoid in pam.medoids]
            self.medoids_indices = new_medoids
            self.update_clusters_assignment()

            # calculate dissimilarity. If this value is less than the current minimum,
            # use this value as the current minimum and update best medoids set
            new_dissimilarity = self.calculate_dissimilarity()
            if new_dissimilarity < self.best_dissimilarity:
                self.best_medoids = new_medoids
                self.best_dissimilarity = new_dissimilarity
