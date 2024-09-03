import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000, degree=10, threshold=0.5):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.degree = degree
        self.weights = None
        self.bias = None
        self.losses = []
        self.accuracies = []  
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.scaler = MinMaxScaler()
        self.threshold = threshold

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_poly_scaled = self.scaler.fit_transform(X_poly)

        m, n = X_poly_scaled.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(self.epochs):
            lin_model = np.dot(X_poly_scaled, self.weights) + self.bias
            y_pred = 1 / (1 + np.exp(-lin_model))

            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)) #1e-15 is added in case y_pred is one to prevent errors :)
            self.losses.append(loss)

            y_pred_bin = np.where(y_pred > self.threshold, 1, 0)
            accuracy = np.mean(y_pred_bin == y)
            self.accuracies.append(accuracy)

            dw = (1 / m) * np.dot(X_poly_scaled.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_poly_scaled = self.scaler.transform(X_poly)

        lin_model = np.dot(X_poly_scaled, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-lin_model))
        y_pred_bin = np.where(y_pred > self.threshold, 1, 0)
        return y_pred_bin