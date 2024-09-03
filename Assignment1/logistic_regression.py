import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=10000, degree=100):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.degree = degree
        self.weights = None
        self.bias = None
        self.losses = []  
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_poly_scaled = self.scaler.fit_transform(X_poly)

        m, n = X_poly_scaled.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(self.epochs):
            lin_model = np.dot(X_poly_scaled, self.weights) + self.bias
            y_pred = 1 / (1 + np.exp(-lin_model))

            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)) #
            self.losses.append(loss)

            dw = (1 / m) * np.dot(X_poly_scaled.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_poly_scaled = self.scaler.transform(X_poly)

        lin_model = np.dot(X_poly_scaled, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-lin_model))
        y_pred_bin = np.where(y_pred > 0.5, 1, 0)
        return y_pred_bin
