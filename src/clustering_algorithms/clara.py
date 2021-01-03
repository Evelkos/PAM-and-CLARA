import random
from statistics import mean

from clustering_algorithms.k_medoids_algorithm import KMedoidsAlgorithm
from clustering_algorithms.pam import PAM


class CLARA(KMedoidsAlgorithm):
    def __init__(self, points, clusters_num=2, labels=None, samples_num=None):
        super().__init__(points=points, clusters_num=clusters_num, labels=labels)

        self.update_clusters_assignment()
        self.best_medoids = self.medoids_indices
        self.best_dissimilarity = self.calculate_dissimilarity()

        # number of samples that will be passed to the PAM algorithm
        if samples_num:
            self.samples_num = samples_num
        else:
            self.samples_num = min(40 + 2 * clusters_num, len(self.points))

    def draw_samples(self):
        """
        Draw a sample of `samples_num` objects randomly from the entire data set.
        """
        return random.sample(self.points, self.samples_num)

    def calculate_dissimilarity(self):
        return mean(
            [point.compute_distance(point.nearest_medoid) for point in self.points]
        )

    def run(self):
        for idx in range(5):
            # draw samples randomly from the entire dataset

            # call PAM to find medoids of the sample
            pam = PAM(
                points=self.draw_samples(),
                clusters_num=self.clusters_num,
                labels=self.labels,
            )
            pam.run()

            # for each point determine the most similar medoid
            new_medoids = [medoid.idx for medoid in pam.medoids]
            self.medoids_indices = new_medoids
            self.update_clusters_assignment()

            # calculate dissimilarity. If this value is less than the current minimum,
            # use this value as the current minimum and update best medoids set
            new_dissimilarity = self.calculate_dissimilarity()
            if new_dissimilarity < self.best_dissimilarity:
                self.best_medoids = new_medoids
                self.best_dissimilarity = new_dissimilarity
