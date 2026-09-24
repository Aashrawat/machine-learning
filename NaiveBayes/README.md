# DEMO OF NAIVE BAYES

Titanic survival classification with scikit-learn’s `GaussianNB`.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $x$ | feature vector | One passenger’s features (age, fare, class, …). |
| $x_j$ | feature | The $j$-th column of $x$. |
| $y$ | label | True class: $1$ = survived, $0$ = did not. |
| $\hat{y}$ | prediction | Class with the highest posterior. |
| $P(y=c)$ | prior | How common class $c$ is in the training set. |
| $P(x \mid y=c)$ | likelihood | How plausible these features are if the class is $c$. |
| $P(y=c \mid x)$ | posterior | Probability of class $c$ after seeing $x$. |
| $\mu_{c,j}$ | class mean | Average of feature $j$ among training rows in class $c$. |
| $\sigma_{c,j}^2$ | class variance | Spread of feature $j$ inside class $c$. |
| $m$ | sample size | Number of training examples. |
| $n$ | features | Number of columns in $x$. |

## What Naive Bayes is

**Naive Bayes** picks the class that is most probable after seeing the features. Bayes’ rule writes that posterior as:

$$
P(y=c \mid x) = \frac{P(x \mid y=c)\, P(y=c)}{P(x)}
$$

$P(x)$ is the same for every class, so the winning class is the one that maximizes the product of **likelihood** and **prior**:

$$
\hat{y} = \arg\max_c \; P(x \mid y=c)\, P(y=c)
$$

The **naive** part is the independence assumption. The model treats each feature as independent of the others once the class is known:

$$
P(x \mid y=c) = \prod_{j=1}^{n} P(x_j \mid y=c)
$$

Age and fare are not really independent, and neither are class and fare. The assumption is often wrong and the classifier can still rank classes well, because it only needs the product to be larger for the correct class.

Multiplying many probabilities underflows, so the fit uses a sum of logs:

$$
\hat{y} = \arg\max_c \left[
  \log P(y=c) + \sum_{j=1}^{n} \log P(x_j \mid y=c)
\right]
$$

`predict` returns that class. `predict_proba` turns the same scores back into probabilities that add up to $1$.

## Gaussian likelihood

Titanic columns in this demo are numbers (age, fare, and integer codes for sex and embarked). **Gaussian Naive Bayes** models each feature inside each class as a bell curve:

$$
P(x_j \mid y=c) =
\frac{1}{\sqrt{2\pi \sigma_{c,j}^2}}
\exp\left(
  -\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2}
\right)
$$

During `fit`, `GaussianNB` estimates:

- $P(y=c)$ from the share of training rows in class $c$ (`class_prior_`)
- $\mu_{c,j}$ from the mean of feature $j$ in that class (`theta_`)
- $\sigma_{c,j}^2$ from the variance of feature $j$ in that class (`var_`)

A small `var_smoothing` term is added to every variance so a feature that never varies inside a class does not make the density explode.

Because the model stores a mean and a variance per feature and per class, this notebook fits `GaussianNB` on the numeric columns directly.

`LabelEncoder` turns `sex` and `embarked` into integers. GaussianNB then treats those codes as numbers on a line and fits a bell curve to them. That is a rough model of a category. It matches the same Titanic table used in the other demos.

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
| `Implementation.ipynb` | Load Titanic, clean/encode features, train/test split, fit Gaussian Naive Bayes, then class means, accuracy, confusion matrix, and classification report. |

Open the notebook on GitHub or in [Colab](https://colab.research.google.com/github/Aashrawat/machine-learning/blob/main/NaiveBayes/Implementation.ipynb).

## Pipeline in the notebook

1. Load `sns.load_dataset("titanic")`.
2. Drop unused columns (`deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`).
3. Fill missing `age` with the mean; drop rows missing `embarked`.
4. `LabelEncoder` on `sex` and `embarked`; cast the table to `int`.
5. Target $y$ = `survived`; $X$ = the other columns.
6. $80/20$ train/test split (`random_state=42`).
7. `GaussianNB` estimates a prior, a mean, and a variance per class.
8. Evaluate with `accuracy_score`, `confusion_matrix`, and `classification_report`.
