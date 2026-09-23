# DEMO OF LOGISTIC REGRESSION

Titanic survival classification with scikit-learn’s `LogisticRegression`.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $x$ | feature vector | One passenger’s inputs (age, fare, class, …). |
| $X$ | design matrix | All training rows stacked. |
| $y$ | label | True class: $1$ = survived, $0$ = did not. |
| $\hat{p}$ | probability | Model’s $P(y=1 \mid x)$. |
| $\hat{y}$ | prediction | Predicted class after a threshold (usually $0.5$). |
| $\theta$ | parameters | Weights the model learns, including intercept. |
| $\sigma(z)$ | sigmoid | Squashes any real $z$ into $(0, 1)$. |
| $m$ | sample size | Number of training examples. |
| $J(\theta)$ | cost | Log loss — how wrong the probabilities are. |

## What logistic regression is

Linear regression predicts a **number**. Logistic regression predicts a **class** by first predicting a **probability**.

The linear score $z = \theta^{\top} x$ can be any real number. The **sigmoid** maps it into a probability:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\hat{p} = \sigma(\theta^{\top} x) = \frac{1}{1 + e^{-\theta^{\top} x}}
$$

- If $\theta^{\top} x$ is large and positive, $\hat{p}$ is close to $1$ (likely survived).
- If $\theta^{\top} x$ is large and negative, $\hat{p}$ is close to $0$ (likely did not).
- If $\theta^{\top} x = 0$, $\hat{p} = 0.5$.

The usual class rule is:

$$
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

In this notebook, `model.predict(X_test)` uses that $0.5$ cutoff. The decision boundary is the set of $x$ where $\theta^{\top} x = 0$.

## Cost function

Squared error is a poor fit for probabilities. Logistic regression minimizes **log loss** (binary cross-entropy):

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[
  y^{(i)} \log \hat{p}^{(i)}
  + (1 - y^{(i)}) \log \big(1 - \hat{p}^{(i)}\big)
\Big]
$$

- If the true label is $1$, only $\log \hat{p}$ matters — predicting $\hat{p}$ near $0$ is expensive.
- If the true label is $0$, only $\log(1-\hat{p})$ matters — predicting $\hat{p}$ near $1$ is expensive.

scikit-learn’s `LogisticRegression.fit` finds $\theta$ that makes $J(\theta)$ small (typically with a regularized solver, not a handwritten gradient loop).

## Model evaluation

Both this notebook and the KNN demo score predictions with a **confusion matrix** and a **classification report**. For Titanic, the **positive** class is survived ($y=1$).

| | Predicted $0$ | Predicted $1$ |
| --- | --- | --- |
| **True $0$** | $TN$ (true negative) | $FP$ (false positive) |
| **True $1$** | $FN$ (false negative) | $TP$ (true positive) |

- **TP**: predicted survived, and they did.
- **FP**: predicted survived, but they did not.
- **TN**: predicted did not survive, and they did not.
- **FN**: predicted did not survive, but they did.

### Accuracy

Fraction of all predictions that are correct:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

That is `accuracy_score(y_test, y_pred)`. Accuracy can look high when one class is common (most passengers did not survive), even if the model misses many survivors.

### Precision

Of the people the model said survived, how many actually did?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

High precision means few false alarms.

### Recall

Of the people who actually survived, how many did the model catch?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

High recall means few missed survivors. Also called **sensitivity** or **true positive rate**.

### F1 score

Harmonic mean of precision and recall. It is high only when **both** are high:

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

`classification_report` prints precision, recall, and $F_1$ for class $0$ and class $1$, plus support (how many true examples of that class).

| Metric | Question it answers |
| --- | --- |
| Accuracy | Overall, how often is $\hat{y}$ right? |
| Precision | When I predict $1$, how often am I right? |
| Recall | Of all true $1$s, how many did I find? |
| $F_1$ | Single score that balances precision and recall. |

## Files

| File | What it shows |
| --- | --- |
| `Implementation.ipynb` | Load Titanic, clean/encode features, train/test split, fit logistic regression, then accuracy, confusion matrix, and classification report. |

Open the notebook on GitHub or in [Colab](https://colab.research.google.com/github/Aashrawat/machine-learning/blob/main/LogisticRegression/Implementation.ipynb).
