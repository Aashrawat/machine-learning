# DEMO OF KNN

Titanic survival classification with scikit-learn’s `KNeighborsClassifier`.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $x$ | feature vector | One passenger’s features (age, fare, class, …). |
| $y$ | label | True class: $1$ = survived, $0$ = did not. |
| $\hat{y}$ | prediction | Majority class among the $k$ nearest neighbors. |
| $k$ | neighbors | How many nearest training points vote. |
| $d(x, x')$ | distance | How far two points are (usually Euclidean). |
| $m$ | sample size | Number of training examples. |
| $n$ | features | Number of columns in $x$. |

## What KNN is

**K-nearest neighbors** does not learn a line or a set of weights. For a new point $x$, it:

1. Measures distance from $x$ to every training example.
2. Keeps the $k$ closest points.
3. Predicts the **majority class** among those neighbors (classification) or their **average** $y$ (regression).

$$
\hat{y} = \text{majority}\big( y^{(i)} \text{ for the } k \text{ nearest } x^{(i)} \big)
$$

Default distance in this notebook is **Euclidean**:

$$
d(x, x') = \sqrt{\sum_{j=1}^{n} (x_j - x'_j)^2}
$$

Because every feature is part of that sum, a column with a large range (fare) can dominate one with a small range (sibsp). That is why this demo **scales** features with `StandardScaler` before fitting KNN: each feature has mean $0$ and variance $1$ after scaling.

KNN stores the training set and looks up neighbors at prediction time (**lazy** learning). Logistic regression instead learns $\theta$ during `fit` and then computes $\sigma(\theta^{\top} x)$.

### Choosing $k$

| $k$ | Effect |
| --- | --- |
| Small (e.g. $1$) | Flexible, can overfit noise. |
| Large | Smoother, can miss local patterns. |
| Odd $k$ for $2$ classes | Avoids tie votes. |

This notebook uses $k = 5$.

## Model evaluation

Both this notebook and the logistic regression demo score predictions with a **confusion matrix** and a **classification report**. For Titanic, the **positive** class is survived ($y=1$).

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

That is `accuracy_score(y_test, y_knn_pred)`. Accuracy can look high when one class is common (most passengers did not survive), even if the model misses many survivors.

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
| `KNNimplementation.ipynb` | Load Titanic, clean/encode features, scale, train/test split, fit KNN ($k=5$), then accuracy, confusion matrix, and classification report. |

Open the notebook on GitHub or in [Colab](https://colab.research.google.com/github/Aashrawat/machine-learning/blob/main/KNN/KNNimplementation.ipynb).

## Pipeline in the notebook

1. Load `sns.load_dataset("titanic")`.
2. Drop unused columns (`deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`).
3. Fill missing `age` with the mean; drop rows missing `embarked`.
4. `LabelEncoder` on `sex` and `embarked`; cast the table to `int`.
5. Target $y$ = `survived`; $X$ = the other columns.
6. $80/20$ train/test split (`random_state=42`).
7. `StandardScaler` on the features, then `KNeighborsClassifier(n_neighbors=5)`.
8. Evaluate with `accuracy_score`, `confusion_matrix`, and `classification_report`.
