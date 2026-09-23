# DEMO OF KNN

Titanic survival classification with scikit-learn’s `KNeighborsClassifier`.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $x$ | feature vector | One passenger’s features (age, fare, class, …). |
| $y$ | label | Class to predict (`survived`: 0 or 1). |
| $k$ | neighbors | How many nearest training points vote. |
| $d(x, x')$ | distance | How far two points are (usually Euclidean). |
| $m$ | sample size | Number of training examples. |

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

Because every feature is part of that sum, a column with a large range (fare) can dominate one with a small range (sibsp). That is why this demo **scales** features with `StandardScaler` before fitting KNN.

### Choosing $k$

| $k$ | Effect |
| --- | --- |
| Small (e.g. 1) | Flexible, can overfit noise. |
| Large | Smoother, can miss local patterns. |
| Odd $k$ for 2 classes | Avoids tie votes. |

This notebook uses $k = 5$.

KNN stores the training set and looks up neighbors at prediction time (**lazy** learning). Logistic regression instead learns $\theta$ during `fit` and then just computes $\sigma(X\theta)$.

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
6. 80/20 train/test split (`random_state=42`).
7. `StandardScaler` on the features, then `KNeighborsClassifier(n_neighbors=5)`.
8. Evaluate with `accuracy_score`, `confusion_matrix`, and `classification_report`.
