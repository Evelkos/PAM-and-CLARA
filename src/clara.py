import random

import numpy as np

from pam import PAM
from utils import compute_distance
from k_medoids_algorithm import KMedoidsAlgorithm
from statistics import mean

class CLARA(KMedoidsAlgorithm):
    def __init__(self, df, clusters_num=2, labels=None, samples_num = None, seed=44):
        super().__init__(df=df, clusters_num=clusters_num, labels=labels, points=None, seed=seed)

        self.update_clusters_assignment()
        self.best_medoids = self.medoids_indices
        self.best_dissimilarity = self.calculate_dissimilarity()

        # number of samples that will be passed to the PAM algorithm
        if samples_num:
            self.samples_num = samples_num
        else:
            self.samples_num = min(40 + 2 * clusters_num, len(self.df))

    def draw_samples(self):
        """
        Draw a sample of `samples_num` objects randomly from the entire data set.
        """
        idx_range = range(len(self.df))
        rows = random.sample(idx_range, self.samples_num)
        return self.df.iloc[rows], [self.points[row] for row in rows]

    def calculate_dissimilarity(self):
        return mean([compute_distance(point.coordinates, point.nearest_medoid.coordinates) for point in self.points])

    def run(self):
        for idx in range(5):
            # draw samples randomly from the entire dataset
            df, points = self.draw_samples()

            # call PAM to find medoids of the sample
            pam = PAM(df, clusters_num=self.clusters_num, labels=self.labels, points=points, seed=self.seed)
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
