import numpy as np
import matplotlib.pyplot as plt


class NaiveBayesClassifier:
    def __init__(self, num_class, num_feature, dist_type):
        self.num_class = num_class
        self.num_feature = num_feature
        self.dist_type = dist_type

        self.prior = np.zeros((num_class, ))
        
        if dist_type == "discrete":
            self.num_bins = 32
            self.num_item_per_bin = 256 / self.num_bins
            self.likelihood = np.zeros((num_class, num_feature, self.num_bins))

    def build_prior(self, y):
        for val in y:
            self.prior[val] += 1
        self.prior = self.prior / y.shape[0]

    def build_likelihood(self, X, y):
        for c in range(self.num_class):
            mask = (y == c)
            X_subset = X[mask]
            # plt.imshow(X_subset[0])
            # plt.show()
            for sample in X_subset:
                features = np.reshape(sample, (-1))
                bins = (features // self.num_item_per_bin).astype("uint8")
                for idx, bin in enumerate(bins):
                    self.likelihood[c, idx, bin] += 1
        
            # nomalize distribution of each pixel
            self.likelihood[c] /= X_subset.shape[0]

            # avoid empty bin of each pixel
            self.likelihood[c] += 1e-5

    def fit(self, X, y):
        self.build_prior(y)
        self.build_likelihood(X, y)
    
    def predict(self, features):
        bins = (features // self.num_item_per_bin).astype("uint8")
        posterior = np.log(self.prior)
        for c in range(self.num_class):
            for idx, bin in enumerate(bins):
                posterior[c] += np.log(self.likelihood[c, idx, bin])
        return posterior

    def evaluate(self, X, y):
        error_count = 0
        idx = 0
        for sample, label in zip(X, y):
            posterior = self.predict(np.reshape(sample, (-1)))
            prediction = np.argmax(posterior)
            if prediction != label:
                error_count += 1
                # print(f"prediction ({idx}): {prediction}")
                # plt.imshow(sample)
                # plt.show()
            
            if (idx+1) % 1000 == 0:
                print(f"prediction on #{idx+1} sample")
            idx += 1

        print(f"error rate: {error_count / y.shape[0]}")


        