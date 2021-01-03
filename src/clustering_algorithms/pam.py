from typing import List, Tuple

from clustering_algorithms.k_medoids_algorithm import KMedoidsAlgorithm
from clustering_algorithms.point import Point


class PAM(KMedoidsAlgorithm):
    def __init__(
        self, points: List[Point], clusters_num: int = 2, labels: List["str"] = None
    ):
        super().__init__(points=points, clusters_num=clusters_num, labels=labels)
        self.medoids = self.prepare_medoids()

    def compute_replacement_cost(self, old_medoid: Point, new_medoid: Point) -> float:
        """
        Compute cost replacing `old_medoid` with `new_medoid`.

        Arguments:
            old_medoid: medoid that we want to replace
            new_medoid: point that has a chance to become a new medoid

        Return:
            Total cost of replacing `old_medoid` with `new_medoid`.

        """
        cost = 0
        for point in self.points:
            if not point == new_medoid and point not in self.medoids:
                cost += point.compute_medoid_replacement_cost(
                    old_medoid, new_medoid, self.medoids
                )

        return cost

    def get_best_replacement_for_medoid(self, old_medoid: Point) -> Tuple[Point, float]:
        """
        Analyse each point from dataset (skip points that are already medoids) and
        compute cost of replacing `old_medoid` with this point. Return the best
        replacement for `old_medoid`.

        Arguments:
            old_medoid: medoid that we want to replace

        Return:
            Tuple with point that is the best replacement for `old_medoid` and
            cost of replacing `old_medoid` with this point.

        """
        replacements = []
        for new_medoid in self.points:
            if new_medoid not in self.medoids:
                cost = self.compute_replacement_cost(old_medoid, new_medoid)
                replacements.append({"cost": cost, "new_medoid": new_medoid})

        best_replacement = min(replacements, key=lambda x: x["cost"])
        return best_replacement["new_medoid"], best_replacement["cost"]

    def run(self) -> None:
        """
        Run PAM algorithm to find the best clusters. Use pam_instance.get_result_df()
        to fetch the results.

        """
        while True:
            # prepare list of currently used medoids
            self.medoids = self.prepare_medoids()
            self.update_clusters_assignment()

            # get the best replacement for each medoid
            replacements = []
            for old_medoid in self.medoids:
                new_medoid, cost = self.get_best_replacement_for_medoid(old_medoid)
                replacements.append((cost, old_medoid, new_medoid))

            # swap (old_medoid, new_medoid) pair, that gives the best result
            best_replacement = min(replacements, key=lambda x: x[0])
            self.swap_medoids(best_replacement[1], best_replacement[2])

            # stop calculations when cost is no longer negative
            if best_replacement[0] >= 0:
                break
