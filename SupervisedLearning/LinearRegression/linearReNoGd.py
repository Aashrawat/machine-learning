import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y = make_regression(n_samples = 100, n_features = 1, noise = 15, random_state =42)
y=y.reshape(-1,1)
m = X.shape[0]
X_b = np.c_[np.ones((m,1)), X]
theta = np.array([[2.0], [3.0]])
plt.figure (figsize=(10,5))
plt.scatter(X, y, color='black', label="Actual Data ")
plt.plot (X, X_b.dot(theta), color='red', label="Initial Line(NO GD)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression without GD")
plt.legend()
plt.show()