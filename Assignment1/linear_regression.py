import numpy as np
import pandas as pd

class LinearRegression():
    
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.X_min = None
        self.X_max = None
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)

        X = self.normalize(X)

        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(self.epochs):
            lin_model = np.dot(X, self.weights) + self.bias

            dw = (1/m) * np.dot(X.T, (lin_model - y))
            db = (1/m) * np.sum(lin_model - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Generates predictions
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        
        X = self.normalize(X)
        
        y_pred = np.dot(X, self.weights) + self.bias
        
        return y_pred
    
    def normalize(self, X):
        X = X.values if isinstance(X, pd.Series) else X     #Trenger denne for å få riktig dimensjoner
        X = X.reshape(-1, 1) if X.ndim == 1 else X          #Denne også
        X = (X - self.X_min) / (self.X_max - self.X_min)
        return X
    
    def printExpressions(self):
        print('Weight: ',self.weights[0], 'Bias: ',self.bias)
        
        
        
        
        print('Et godt uttrykk for Energy Consumption er: ',round(self.weights[0],4),'* x +',round(self.bias,4))


    