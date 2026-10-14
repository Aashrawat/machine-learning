## machine-learning

### Supervised learning

Labeled examples and a target to predict live in [`SupervisedLearning/`](SupervisedLearning/).

| Folder | Topic |
| --- | --- |
| [`SupervisedLearning/LinearRegression/`](SupervisedLearning/LinearRegression/) | Linear regression and gradient descent |
| [`SupervisedLearning/LinearRegressionPrj/`](SupervisedLearning/LinearRegressionPrj/) | Ford used-car prices: EDA, one-hot vs label encoding, linear regression |
| [`SupervisedLearning/LogisticRegression/`](SupervisedLearning/LogisticRegression/) | Titanic survival with logistic regression |
| [`SupervisedLearning/KNN/`](SupervisedLearning/KNN/) | Titanic survival with k-nearest neighbors |
| [`SupervisedLearning/NaiveBayes/`](SupervisedLearning/NaiveBayes/) | Titanic survival with Gaussian Naive Bayes |
| [`SupervisedLearning/DecisionTree/`](SupervisedLearning/DecisionTree/) | Titanic survival with a decision tree |
| [`SupervisedLearning/RandomForest/`](SupervisedLearning/RandomForest/) | Random forest |
| [`SupervisedLearning/EnsembleLearning/`](SupervisedLearning/EnsembleLearning/) | Ensemble learning |
| [`SupervisedLearning/SVC/`](SupervisedLearning/SVC/) | Titanic survival with an RBF support vector classifier |
| [`SupervisedLearning/CrossValidation/`](SupervisedLearning/CrossValidation/) | 5-fold cross-validation, then `GridSearchCV` over an SVC |

- **[LinearRegression/README.md](SupervisedLearning/LinearRegression/README.md)** — equations, symbol table, and how gradient descent fits the line.
- **[LinearRegressionPrj/README.md](SupervisedLearning/LinearRegressionPrj/README.md)** — Ford listings: attribute notes, EDA, one-hot vs `LabelEncoder`, scaling, and linear regression (one-hot ~0.85 R² vs label ~0.74).
- **[LogisticRegression/README.md](SupervisedLearning/LogisticRegression/README.md)** — sigmoid, log loss, then accuracy, precision, recall, and $F_1$.
- **[KNN/README.md](SupervisedLearning/KNN/README.md)** — Euclidean neighbors and $k$, then the same evaluation metrics.
- **[NaiveBayes/README.md](SupervisedLearning/NaiveBayes/README.md)** — prior, Gaussian likelihood, and the independence assumption, then accuracy, precision, recall, and $F_1$.
- **[DecisionTree/README.md](SupervisedLearning/DecisionTree/README.md)** — Gini splits and leaf majority votes, then accuracy, precision, recall, and $F_1$.
- **[SVC/README.md](SupervisedLearning/SVC/README.md)** — maximum-margin boundary and the RBF kernel, then accuracy, precision, recall, and $F_1$.
- **[CrossValidation/README.md](SupervisedLearning/CrossValidation/README.md)** — 5-fold scores and their mean, then `GridSearchCV` over `C` and kernel. The same search works for linear regression, logistic regression, KNN, and Naive Bayes.

### Other

| Folder | Topic |
| --- | --- |
| [`EDAProject1/`](EDAProject1/) | EDA, data cleaning, preprocessing, feature engineering and extraction |
| [`K-meanClustering/`](K-meanClustering/) | K-means clustering |
| [`DBScan/`](DBScan/) | DBSCAN clustering |
| [`NLP/`](NLP/) | Natural language processing |

- **[EDAProject1/README.md](EDAProject1/README.md)** — insurance charges: explore, clean, encode, engineer BMI/region features, then select with Pearson and chi-square.
