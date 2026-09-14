import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y = make_regression(n_samples = 100, n_features = 1, noise = 15, random_state =42)
y=y.reshape(-1,1)
m = X.shape[0]
X_b = np.c_[np.ones((m,1)), X]
theta = np.array([[2.0], [3.0]])

learning_rate = 0.1
n_iterations = 100

for _ in range(n_iterations):

    y_pred = X_b.dot(theta)

    gradients = (2 / m) * X_b.T.dot(y_pred - y)

    theta -= learning_rate * gradients

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, X_b.dot(theta), color="red", label="Optimized Line (With GD)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression With Gradient Descent")
plt.legend()
plt.show()