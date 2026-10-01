# DEMO OF CROSS VALIDATION

Titanic survival with an RBF support vector classifier, scored with **5-fold cross-validation**.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $X$ | features | Passenger columns after cleaning (age, fare, class, …). |
| $y$ | label | True class: $1$ = survived, $0$ = did not. In the notebook this vector is `Y`. |
| $k$ | folds | How many pieces the data is split into. This notebook uses $k = 5$. |
| $\text{score}_i$ | fold score | Accuracy on fold $i$ after training on the other folds. |
| $\text{CV}$ | cross-validation score | Mean of the $k$ fold scores. |

## What cross-validation is

One train/test split can land the easy passengers in the test set, or the hard ones. The accuracy from that single cut then looks better or worse than the model usually is.

**k-fold cross-validation** repeats the test:

1. Cut the rows into $k$ folds of about the same size.
2. Hold out fold $i$. Train on the other $k - 1$ folds.
3. Score the held-out fold.
4. Repeat until every fold has been the test set once.
5. Average those $k$ scores.

$$
\text{CV} = \frac{1}{k} \sum_{i=1}^{k} \text{score}_i
$$

`cross_val_score(estimator, X, y, cv=5, scoring="accuracy")` runs that loop. Each fold trains a fresh copy of the estimator. Passing the already-fit `SVC` does not reuse the earlier 80/20 fit; scikit-learn clones the model and fits the clone on that fold’s training rows.

This notebook sets `cv=5` and `scoring="accuracy"`, then prints the five fold accuracies and their mean.

The notebook scales the full feature matrix before `cross_val_score`, so the scaler’s mean and variance include rows that a fold will later treat as test data.

## Cross-validation works on every model

The model is only the first argument. The same call works for the other models in this repo. Classification models can keep `scoring="accuracy"`. Linear regression predicts a number, so it uses a regression score such as `"r2"`.

| Model | Estimator | Typical `scoring` |
| --- | --- | --- |
| Linear regression | `LinearRegression()` | `"r2"` |
| Logistic regression | `LogisticRegression()` | `"accuracy"` |
| K-nearest neighbors | `KNeighborsClassifier()` | `"accuracy"` |
| Naive Bayes | `GaussianNB()` | `"accuracy"` |
| Decision tree | `DecisionTreeClassifier()` | `"accuracy"` |
| SVC (this notebook) | `SVC(kernel="rbf")` | `"accuracy"` |

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
cross_val_score(LogisticRegression(), X, y, cv=5, scoring="accuracy")
cross_val_score(KNeighborsClassifier(), X, y, cv=5, scoring="accuracy")
cross_val_score(GaussianNB(), X, y, cv=5, scoring="accuracy")
```

For a classifier, `scoring` can also be `"precision"`, `"recall"`, or `"f1"`. Those are the same metrics as in the logistic regression, KNN, Naive Bayes, decision tree, and SVC notebooks. Only the way the score is averaged changes: one held-out split versus $k$ of them.

## Files

| File | What it shows |
| --- | --- |
| `CrossValidation_SVM.ipynb` | Load Titanic, clean and encode features, fit an RBF `SVC` on an 80/20 split, then 5-fold `cross_val_score` accuracy on the scaled features. |

Open the notebook on GitHub or in [Colab](https://colab.research.google.com/github/Aashrawat/machine-learning/blob/main/CrossValidation/CrossValidation_SVM.ipynb).

## Pipeline in the notebook

1. Load `sns.load_dataset("titanic")`.
2. Drop unused columns (`deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`).
3. Fill missing `age` with the mean; drop rows missing `embarked`.
4. `LabelEncoder` on `sex` and `embarked`; cast the table to `int`.
5. Target `Y` = `survived`; `X` = the other columns.
6. $80/20$ train/test split (`random_state=42`).
7. `StandardScaler` on the split, then `SVC(kernel="rbf")` and `predict` on the test set.
8. Scale the full feature matrix `X`.
9. `cross_val_score(..., cv=5, scoring="accuracy")`, then print the five scores and their mean.
