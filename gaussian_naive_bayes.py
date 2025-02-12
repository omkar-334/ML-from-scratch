import numpy as np

# Sources
# https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
# https://github.com/scikit-learn/scikit-learn/blob/6a0838c41/sklearn/naive_bayes.py#L151
# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/


class GaussianNB:
    def __init__(self):
        self.parameters = {}
        self.classes = []
        self.likelihoods = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            curr_cls = X[y == cls]
            self.parameters[cls] = {
                "mean": np.mean(curr_cls, axis=0),
                "var": np.var(curr_cls, axis=0),
                # prior - Probability(Curr) - Shape of current class divided by entire dataset
                "prior": curr_cls.shape[0] / X.shape[0],
            }

    def calculate_likelihood(self, cls, x):
        var = self.parameters[cls]["var"]
        mean = self.parameters[cls]["mean"]
        denominator = np.sqrt(var * 2 * np.pi)
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        return numerator / denominator

    def calculate_posterior(self, x):
        posteriors = []
        for cls in self.classes:
            prior = np.log(self.parameters[cls]["prior"])
            likelihood = np.sum(np.log(self.calculate_likelihood(cls, x)))
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self.calculate_posterior(x) for x in X]
        return np.array(y_pred)
