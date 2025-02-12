import numpy as np


def mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))


# Sources
# https://github.com/mahdi-eth/Linear-Regression-from-Scratch
# https://www.kdnuggets.com/linear-regression-from-scratch-with-numpy
# https://medium.com/@pritioli/implementing-linear-regression-from-scratch-747343634494


class LinearRegression:
    def __init__(self, num_iterations=1000, lr=None, lr_scheduler=None):
        self.lr = lr
        self.num_iterations = num_iterations
        self.theta = None
        self.bias = None
        self.tolerance = 1e-5
        self.lr_scheduler = lr_scheduler

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        self.bias = 0

        prev_loss = 0

        for epoch in range(self.num_iterations):
            y_pred = np.dot(X, self.theta) + self.bias
            error = y_pred - y

            dw = (1 / n_samples) * (np.dot(X.T, (error)))
            db = (1 / n_samples) * (np.sum(error))

            if self.lr_scheduler:
                self.lr = self.lr_scheduler.get_lr(epoch, total_epochs=self.num_iterations)

            self.theta -= self.lr * dw
            self.bias -= self.lr * db

            curr_loss = mse(y_pred, y)

            if abs(curr_loss - prev_loss) < self.tolerance:
                break

            prev_loss = curr_loss

    def predict(self, X):
        return np.dot(X, self.theta) + self.bias
