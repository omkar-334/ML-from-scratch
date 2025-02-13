import numpy as np

# Sources
# https://en.wikipedia.org/wiki/Least_squares#Solving_the_least_squares_problem
# https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/


class NormalLinearRegression:
    def __init__(self, fit_intercept=True, ridge_param=0.0):
        self.theta = None
        self.fit_intercept = fit_intercept
        self.ridge_param = ridge_param

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones for the intercept

        lhs = np.linalg.inv(np.dot(X.T, X) + self.ridge_param * np.eye(X.shape[1]))
        rhs = np.dot(X.T, y)
        self.theta = np.dot(lhs, rhs)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones for the intercept
        return np.dot(X, self.theta)
