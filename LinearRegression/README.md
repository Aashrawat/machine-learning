# Linear Regression and Gradient Descent

Small Python demos of **linear regression**: fitting a line with scikit-learn, drawing a bad initial guess, and then improving that guess with **gradient descent**.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $x$ | feature | Input value (for example, lot size). |
| $X$ | design matrix | All features, with a column of $1$s for the intercept. |
| $y$ | target | True output (for example, house price). |
| $\hat{y}$ | prediction | Model’s guess for $y$. |
| $\theta$ | parameters | Vector of weights the model learns. |
| $\theta_0$ | intercept | Predicted $y$ when $x = 0$. |
| $\theta_1$ | slope | How much $\hat{y}$ changes when $x$ increases by $1$. |
| $m$ | sample size | Number of training examples. |
| $J(\theta)$ | cost | Mean squared error — how wrong $\theta$ currently is. |
| $\nabla J(\theta)$ | gradient | Slope of the cost; points uphill. |
| $\alpha$ | learning rate | Step size for each gradient descent update. |
| $\sum$ | sum | Add over all training examples $i = 1 \ldots m$. |

## What linear regression is

Linear regression predicts a continuous target $y$ from features $x$ by assuming the relationship is a straight line (or a hyperplane in higher dimensions):

$$
\hat{y} = \theta_0 + \theta_1 x
$$

- $\theta_0$ is the **intercept** (the predicted value when $x = 0$).
- $\theta_1$ is the **slope** (how much $\hat{y}$ changes when $x$ increases by $1$).
- Together these parameters are often written as a vector $\theta$.

With a column of ones added to $X$ (the bias term), the same model is just a matrix multiply:

$$
\hat{y} = X\theta
$$

That is the form used in `linearRegressionGd.py` and `linearReNoGd.py`.

The goal is to choose $\theta$ so predictions stay close to the true labels. Closeness is measured with **mean squared error (MSE)**:

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

MSE is a bowl-shaped (convex) surface for linear regression, so there is a single global minimum. You can reach that minimum in two common ways:

1. **Closed form (normal equation / OLS)** — solve for $\theta$ directly. scikit-learn’s `LinearRegression` does this kind of fit in `linearRegression.py`.
2. **Gradient descent** — start from a guess and walk downhill on $J(\theta)$ until the error stops improving. That is what `linearRegressionGd.py` does.

## Gradient descent

Gradient descent uses the slope of the cost function. The gradient $\nabla J(\theta)$ points uphill, so you move the other way:

$$
\theta \leftarrow \theta - \alpha \, \nabla J(\theta)
$$

- $\alpha$ is the **learning rate**. Too small and training crawls; too large and $\theta$ can overshoot and diverge.
- Each step uses all $m$ training examples (**batch** gradient descent).

For MSE, the gradient of $J$ (using the $2/m$ convention that matches this code) is:

$$
\nabla J(\theta) = \frac{2}{m} X^{\top} (X\theta - y)
$$

That is exactly this loop:

```python
y_pred = X_b.dot(theta)
gradients = (2 / m) * X_b.T.dot(y_pred - y)
theta -= learning_rate * gradients
```

Intuition:

- If a prediction is too high, $y_{\text{pred}} - y$ is positive, so $\theta$ is nudged down.
- If a prediction is too low, $\theta$ is nudged up.
- Repeating this many times moves the red line onto the cloud of points.

`linearReNoGd.py` plots the **same initial $\theta$** with no updates, so you can see how far off a random starting line can be. `linearRegressionGd.py` then runs $100$ steps at $\alpha = 0.1$ and plots the fitted line.

### Practical notes

| Choice | What it does |
| --- | --- |
| Learning rate $\alpha$ | Step size. Tune it if the line blows up or barely moves. |
| Iterations | How many downhill steps. Stop when $J(\theta)$ plateaus. |
| Feature scale | GD is much happier when features are on similar scales. |
| Batch vs SGD vs mini-batch | Full-batch is stable; stochastic / mini-batch is noisier but scales to huge datasets. |

Linear regression does **not** need gradient descent when $X^{\top}X$ is small and invertible — the normal equation is faster and exact. GD still matters because:

- it scales to large $m$ or many features
- the same idea trains logistic regression, neural nets, and most modern models
- it is the first place you see learning rate, initialization, and iteration count in action

## Files

| File | What it shows |
| --- | --- |
| `linearRegression.py` | House price vs lot size with scikit-learn’s closed-form fit (`Housing.csv`). |
| `linearReNoGd.py` | Synthetic data with a fixed, untrained line. |
| `linearRegressionGd.py` | The same synthetic data after batch gradient descent. |
| `Housing.csv` | Dataset used by the scikit-learn demo. |

## Run the demos

From this folder, with NumPy, pandas, matplotlib, and scikit-learn installed:

```bash
python linearRegression.py
python linearReNoGd.py
python linearRegressionGd.py
```

Compare the untrained line (`linearReNoGd.py`) with the optimized line (`linearRegressionGd.py`) to see gradient descent do its job.
