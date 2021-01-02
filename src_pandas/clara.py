from pam import PAM


class Clara:
    def __init__(self, df, clusters_num=2, labels=None, samples_num = None, seed=44):
        random.seed(seed)
        # this seed will be passed to the PAM algorithm
        self.seed = seed

        self.df = df
        self.clusters_num = clusters_num
        self.labels = labels

        # number of samples that will be passed to the PAM algorithm
        self.samples_num = samples_num

    def draw_samples(self):
        """
        Draw a sample of `samples_num` objects randomly from the entire data set.
        """
        idx_range = range(len(self.df))
        rows = random.sample(idx_range, self.samples_num)

        return self.df.loc[rows]

    def run(self):
        for idx in range(5):
            samples = self.draw_samples()
            pam = PAM(samples, self.clusters_num, self.labels, seed=self.seed)
