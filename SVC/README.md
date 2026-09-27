# DEMO OF SVC

Titanic survival classification with scikit-learn’s `SVC` and an RBF kernel.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $x$ | feature vector | One passenger’s features (age, fare, class, …). |
| $y$ | label | True class: $1$ = survived, $0$ = did not. |
| $\hat{y}$ | prediction | Class on the positive side of the decision function. |
| $K(x, x')$ | kernel | Similarity between two passengers. |
| $\gamma$ | kernel width | How fast RBF similarity falls off with distance. |
| $C$ | soft-margin cost | How hard the fit punishes points on the wrong side of the margin. |
| $\alpha_i$ | dual weight | How much training row $i$ pulls the boundary. Zero except for support vectors. |
| $b$ | intercept | Shift of the decision function. |
| $m$ | sample size | Number of training examples. |
| $n$ | features | Number of columns in $x$. |

## What an SVC is

A **support vector classifier** draws a boundary between the two classes and keeps that boundary as far as possible from the nearest training points. Those nearest points are the **support vectors**. The other training rows sit farther inside their class and do not move the boundary.

In a space where a straight cut is enough, the score is a hyperplane:

$$
f(x) = w^{\top} x + b
$$

The fit looks for a wide **margin** around that cut. A soft margin allows some points to land on the wrong side or inside the margin, with cost $C$ per violation. A large $C$ hugs the training labels more tightly. A small $C$ leaves a wider, simpler margin. This notebook does not set $C$, so it stays at the default $C = 1$.

Titanic passengers are not split by one straight cut in the raw features. The notebook sets `kernel='rbf'`, so the boundary can bend. The RBF kernel scores similarity by squared distance:

$$
K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)
$$

Nearby passengers get a kernel value near $1$. Far passengers get a value near $0$. The model never builds the curved coordinates explicitly. It only needs these pairwise scores (the **kernel trick**). The decision function is a weighted sum over the support vectors:

$$
f(x) = \sum_{i \in SV} y_i \alpha_i \, K(x_i, x) + b
$$

In that sum the class codes are $+1$ and $-1$. The notebook’s `survived` column stays $0$ and $1$; `SVC` maps those labels internally. `predict` returns the original class on the matching side of $f(x)$. `gamma` is left at the default `'scale'`:

$$
\gamma = \frac{1}{n \cdot \mathrm{Var}(X)}
$$

A larger $\gamma$ makes each support vector influence only its close neighbors, which can overfit. A smaller $\gamma$ smooths the boundary.

## Scaling

The notebook runs `StandardScaler` before `fit`. The RBF kernel uses Euclidean distance, so a column with a large range (fare) would dominate one with a small range (sibsp) if the columns stayed in raw units. After scaling, each feature has mean $0$ and variance $1$, and distance treats them on the same scale. Unlike a decision tree, this scaling changes which passengers the kernel treats as neighbors.

## Model evaluation

This notebook scores predictions with a **confusion matrix** and a **classification report**. For Titanic, the **positive** class is survived ($y=1$).

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

That is `accuracy_score(Y_test, y_pred_svc)`. Accuracy can look high when one class is common (most passengers did not survive), even if the model misses many survivors.

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
| `ImplementationSVC.ipynb` | Load Titanic, clean/encode features, scale, train/test split, fit an RBF support vector classifier, then accuracy, confusion matrix, and classification report. |

Open the notebook on GitHub or in [Colab](https://colab.research.google.com/github/Aashrawat/machine-learning/blob/main/SVC/ImplementationSVC.ipynb).

## Pipeline in the notebook

1. Load `sns.load_dataset("titanic")`.
2. Drop unused columns (`deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`).
3. Fill missing `age` with the mean; drop rows missing `embarked`.
4. `LabelEncoder` on `sex` and `embarked`; cast the table to `int`.
5. Target $y$ = `survived`; $X$ = the other columns.
6. $80/20$ train/test split (`random_state=42`).
7. `StandardScaler` on the features, then `SVC(kernel='rbf')`.
8. Evaluate with `accuracy_score`, `confusion_matrix`, and `classification_report`.
