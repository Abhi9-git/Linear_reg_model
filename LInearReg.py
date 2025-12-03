import numpy as np

class LinearRegressionFromScratch:

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Number of samples
        n = len(X)

        # Calculating slope (m)
        self.m = (n * np.sum(X*y) - np.sum(X) * np.sum(y)) / \
                 (n * np.sum(X**2) - (np.sum(X))**2)

        # Calculating intercept (b)
        self.b = (np.sum(y) - self.m * np.sum(X)) / n

    def predict(self, X):
        X = np.array(X)
        return self.m * X + self.b


X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegressionFromScratch()
model.fit(X, y)

print("Slope (m):", model.m)
print("Intercept (b):", model.b)

print("Prediction for X=6:", model.predict([6])[0])
