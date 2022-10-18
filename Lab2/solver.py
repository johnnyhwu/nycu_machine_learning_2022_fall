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
        elif dist_type == "continuous":
            self.likelihood_mean = np.zeros((num_class, num_feature))
            self.likelihood_variance = np.zeros((num_class, num_feature))
        else:
            raise Exception("Unknown distribution type: {dist_type}")

    def build_prior(self, y):
        for val in y:
            self.prior[val] += 1
        self.prior = self.prior / y.shape[0]

    def build_likelihood(self, X, y):
        if self.dist_type == "discrete":
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
        else:
            # calculate mean and variance for each pixel
            for c in range(self.num_class):
                mask = (y == c)
                X_subset = X[mask]
                X_subset_mean = np.mean(X_subset, axis=0)
                X_subset_variance = np.sum((X_subset - X_subset_mean)**2, axis=0) / X_subset.shape[0]
                self.likelihood_mean[c] = np.reshape(X_subset_mean, (-1))
                self.likelihood_variance[c] = np.reshape(X_subset_variance, (-1))


    def fit(self, X, y):
        self.build_prior(y)
        self.build_likelihood(X, y)
    
    def predict(self, features):
        posterior = np.log(self.prior)

        if self.dist_type == "discrete":
            bins = (features // self.num_item_per_bin).astype("uint8")
            for c in range(self.num_class):
                for idx, bin in enumerate(bins):
                    posterior[c] += np.log(self.likelihood[c, idx, bin])
        else:
            for c in range(self.num_class):
                mask = (self.likelihood_variance[c] != 0)
                tmp1 = np.log(2.0 * np.pi * self.likelihood_variance[c][mask]) / (-2.0)
                tmp2 = (-1.0) * ((features[mask] - self.likelihood_mean[c][mask])**2) / (2.0 * self.likelihood_variance[c][mask])
                posterior[c] += np.sum(tmp1+tmp2)

        return posterior

    def evaluate(self, X, y, show_info=False):
        error_count = 0
        idx = 0
        for sample, label in zip(X, y):
            posterior = self.predict(np.reshape(sample, (-1)))
            posterior *= -1
            posterior /= np.sum(posterior)
            prediction = np.argmin(posterior)
            if prediction != label:
                error_count += 1
                # print(f"prediction ({idx}): {prediction}")
                # plt.imshow(sample)
                # plt.show()
            
            if show_info:
                print(f"{'='*10} (#{idx+1} Sample) Posterior {'='*10}")
                for i, v in enumerate(posterior):
                    print(f"{i} => {v}")
                print(f"Prediction: {prediction}, Ans: {label}")
                print()
            else:
                if (idx+1) % 1000 == 0:
                    print(f"prediction on #{idx+1} sample")
            
            idx += 1

        print(f"error rate: {error_count / y.shape[0]}")
    
    def distill(self):
        for c in range(self.num_class):
            print(f"{'='*10} Digit {c} Imagination {'='*10}")

            if self.dist_type == "discrete":
                pixel_dist = self.likelihood[c]
            else:
                pixel_dist = self.likelihood_mean[c]

            arr = []
            for idx, row in enumerate(pixel_dist):
                if self.dist_type == "discrete":
                    white = np.sum(row[0:int(self.num_bins/2)])
                    black = np.sum(row[int(self.num_bins/2):])
                    if white > black:
                        prt_str = "0"
                    else:
                        prt_str = "1"
                else:
                    if row < 128:
                        prt_str = "0"
                    else:
                        prt_str = "1"

                if idx != 0 and (idx+1) % 28 == 0:
                    end_str = "\n"
                else:
                    end_str = " "
                
                print(prt_str, end=end_str)
                arr.append(int(prt_str))
            
            print()

            plt.imshow(np.array(arr).reshape(28, 28))
            plt.show()


            


        