# DEMO OF DECISION TREE

Titanic survival classification with scikit-learn’s `DecisionTreeClassifier`.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $x$ | feature vector | One passenger’s features (age, fare, class, …). |
| $x_j$ | feature | The $j$-th column of $x$. |
| $y$ | label | True class: $1$ = survived, $0$ = did not. |
| $\hat{y}$ | prediction | Majority class in the leaf the passenger reaches. |
| $S$ | node | The training rows that reached one split. |
| $p_c$ | class share | Fraction of rows in $S$ whose label is class $c$. |
| $t$ | threshold | Cut on one feature: go left if $x_j \le t$, right otherwise. |
| $I(S)$ | impurity | How mixed the labels in $S$ are (Gini in this notebook). |
| $m$ | sample size | Number of training examples. |
| $n$ | features | Number of columns in $x$. |

## What a decision tree is

A **decision tree** asks a sequence of yes/no questions about the features. Each question is a threshold on one column, for example “is age $\le 30$?”. Rows that pass go left; the rest go right. The questions continue until a **leaf**, and the leaf’s prediction is the **majority class** of the training rows that landed there.

$$
\hat{y} = \text{majority}\big( y^{(i)} \text{ among rows in the leaf} \big)
$$

The tree is built from the training set only. At each node the fit searches over features $j$ and thresholds $t$ and keeps the split that makes the two child groups as pure as possible.

## Gini impurity

This notebook uses `DecisionTreeClassifier` with the default criterion, **Gini impurity**:

$$
I(S) = 1 - \sum_{c} p_c^2
$$

- If every row in $S$ has the same label, $I(S) = 0$ (a pure node).
- If two classes are split $50/50$, $I(S) = 0.5$ (as mixed as a two-class node gets).

A candidate split sends $|S_L|$ rows left and $|S_R|$ rows right. The score is how much impurity drops, weighted by how many rows each child keeps:

$$
\Delta = I(S) - \left(
  \frac{|S_L|}{|S|} I(S_L) + \frac{|S_R|}{|S|} I(S_R)
\right)
$$

`fit` picks the $j$ and $t$ with the largest $\Delta$, then repeats on each child. With the defaults used here (`max_depth=None`, `min_samples_split=2`), splitting stops when a node is pure or has too few rows to split. That tree can memorize the training set, so test accuracy can lag training accuracy.

An alternative criterion is **entropy**, $H(S) = -\sum_c p_c \log_2 p_c$. This notebook does not set `criterion`, so it stays on Gini.

## Scaling

The notebook runs `StandardScaler` before `fit`, the same step as the KNN demo. A tree only compares each feature with a threshold, so stretching a column by a positive constant does not change which rows go left or right. The learned cuts move with the scaled units; the partition of passengers does not.

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

That is `accuracy_score(Y_test, y_pred_dt)`. Accuracy can look high when one class is common (most passengers did not survive), even if the model misses many survivors.

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
| `ImplementationDecisionTree.ipynb` | Load Titanic, clean/encode features, scale, train/test split, fit a decision tree, then accuracy, confusion matrix, and classification report. |

Open the notebook on GitHub or in [Colab](https://colab.research.google.com/github/Aashrawat/machine-learning/blob/main/DecisionTree/ImplementationDecisionTree.ipynb).

## Pipeline in the notebook

1. Load `sns.load_dataset("titanic")`.
2. Drop unused columns (`deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`).
3. Fill missing `age` with the mean; drop rows missing `embarked`.
4. `LabelEncoder` on `sex` and `embarked`; cast the table to `int`.
5. Target $y$ = `survived`; $X$ = the other columns.
6. $80/20$ train/test split (`random_state=42`).
7. `StandardScaler` on the features, then `DecisionTreeClassifier(random_state=42)`.
8. Evaluate with `accuracy_score`, `confusion_matrix`, and `classification_report`.
