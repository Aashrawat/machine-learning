# DEMO OF NATURAL LANGUAGE PROCESSING

Classify the emotion of a short English sentence with scikit-learn. The notebook cleans the text, turns it into word counts and TF-IDF weights, then compares Multinomial Naive Bayes with logistic regression.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $d$ | document | One sentence. |
| $t$ | term | One word in the vocabulary. |
| $y$ | label | True emotion. |
| $\hat{y}$ | prediction | Emotion the model picks. |
| $V$ | vocabulary | Distinct words kept after cleaning. |
| $c(t, d)$ | term count | Times word $t$ appears in $d$. |
| $\mathrm{df}(t)$ | document frequency | How many training sentences contain $t$. |
| $n$ | sample size | Number of sentences (16,000 in `train.txt`). |
| $P(y=c)$ | prior | How common emotion $c$ is in the training set. |
| $P(t \mid y=c)$ | word likelihood | How typical word $t$ is inside emotion $c$. |

## The data

`train.txt` has 16,000 lines. Each line is a sentence and an emotion, separated by `;`:

```
i didnt feel humiliated;sadness
```

There are no missing values. The notebook maps emotions to integers in the order they first appear:

| Code | Emotion |
| --- | --- |
| $0$ | sadness |
| $1$ | anger |
| $2$ | love |
| $3$ | surprise |
| $4$ | fear |
| $5$ | joy |

An $80/20$ split (`random_state=42`) gives 12,800 training sentences and 3,200 test sentences.

## Cleaning the text

Capitals, punctuation, digits, and words such as "the" appear in every emotion. The notebook strips them before counting words.

| Step | What it does |
| --- | --- |
| Lowercase | `i Feel Sad` and `i feel sad` become the same tokens. |
| Drop punctuation | `string.punctuation` is removed with `str.translate`. |
| Drop digits | Characters for which `isdigit()` is true are removed. |
| Drop non-ASCII | Emoji and other non-ASCII characters are removed with `isascii()`. |
| Drop stopwords | NLTK’s English stopword list (198 words) is removed. The sentence is split on spaces. |

Example from the notebook, row 1:

- before: `i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake`
- after: `go feeling hopeless damned hopeful around someone cares awake`

`word_tokenize` is imported. The stopword step uses `split()`, so punctuation is already gone and does not stay glued to a word.

## Bag of words

`CountVectorizer` builds a column for every word that appears in the training sentences. A sentence becomes a row of counts:

$$
x_d = \big(c(t_1, d),\; c(t_2, d),\; \ldots,\; c(t_{|V|}, d)\big)
$$

Word order is discarded, so "not happy" and "happy not" share the same vector. The test set uses the **training** vocabulary. A word that appears only in the test set is left out.

## TF-IDF

Raw counts give the same weight to a word in almost every sentence and to a word that shows up in one emotion. **TF-IDF** down-weights words that appear in many sentences.

With scikit-learn’s defaults (`smooth_idf=True`, then an L2 norm):

$$
\mathrm{tf}(t, d) = c(t, d)
$$

$$
\mathrm{idf}(t) = \log\frac{1+n}{1+\mathrm{df}(t)} + 1
$$

$$
\mathrm{tfidf}(t, d) = \mathrm{tf}(t, d) \cdot \mathrm{idf}(t)
$$

Each sentence vector is then divided by its L2 length, so a long sentence stays comparable to a short one. `TfidfVectorizer` fits idf on the training split, then transforms the test split.

## Multinomial Naive Bayes

**Multinomial Naive Bayes** treats a sentence as a bag of word draws. It uses the same Bayes rule as the Gaussian Naive Bayes demo, with a likelihood built for counts. Counts and TF-IDF weights are non-negative, which is what `MultinomialNB` expects.

$$
\hat{y} = \arg\max_c \; P(d \mid y=c)\, P(y=c)
$$

The document likelihood is a product over words:

$$
P(d \mid y=c) \propto \prod_{t \in V} P(t \mid y=c)^{c(t, d)}
$$

Each word probability is that word’s share inside class $c$. Laplace smoothing ($\alpha = 1$) keeps a word that never appears in an emotion from getting probability $0$:

$$
P(t \mid y=c) = \frac{N_{c,t} + 1}{N_c + |V|}
$$

$N_{c,t}$ is the total count of word $t$ in training sentences of emotion $c$. $N_c$ is the total of all word counts in that emotion.

The fit adds log probabilities so the product does not underflow:

$$
\hat{y} = \arg\max_c \left[
  \log P(y=c) + \sum_{t \in V} c(t, d)\, \log P(t \mid y=c)
\right]
$$

## Logistic regression on TF-IDF

Logistic regression learns a weight vector $\theta_c$ for each emotion. The score for a sentence is the dot product of its TF-IDF vector with that weight vector. The predicted emotion is the largest score:

$$
\hat{y} = \arg\max_c \; \theta_c^{\top} x
$$

`LogisticRegression(max_iter=1000)` is fit on the same TF-IDF matrix. The higher iteration cap lets the solver finish on this wide, sparse matrix.

## Test accuracy

The notebook prints accuracy on the 3,200 held-out sentences:

| Features | Model | Accuracy |
| --- | --- | --- |
| Bag of words (`CountVectorizer`) | `MultinomialNB` | 0.768 |
| TF-IDF (`TfidfVectorizer`) | `MultinomialNB` | 0.661 |
| TF-IDF (`TfidfVectorizer`) | `LogisticRegression` | 0.863 |

Bag-of-words counts match the multinomial likelihood, so Naive Bayes scores higher on counts (0.768) than on TF-IDF (0.661). Logistic regression treats TF-IDF as features and reaches 0.863, the highest of the three.

## Files

| File | What it shows |
| --- | --- |
| `NLP_ML.ipynb` | Load `train.txt`, clean text, encode emotions, split 80/20, then bag-of-words Naive Bayes, TF-IDF Naive Bayes, and TF-IDF logistic regression. |
| `train.txt` | 16,000 sentences labeled sadness, anger, love, surprise, fear, or joy. |

Open the notebook on GitHub or in [Colab](https://colab.research.google.com/github/Aashrawat/machine-learning/blob/main/NLP/NLP_ML.ipynb).

## Pipeline in the notebook

1. Read `train.txt` with separator `;` and columns `text`, `emotion`.
2. Map the six emotion names to integers $0$–$5$.
3. Lowercase, then remove punctuation, digits, non-ASCII characters, and English stopwords.
4. `train_test_split` with `test_size=0.2` and `random_state=42`.
5. `CountVectorizer` + `MultinomialNB`, then `accuracy_score`.
6. `TfidfVectorizer` + `MultinomialNB`, then `accuracy_score`.
7. The same TF-IDF matrix with `LogisticRegression(max_iter=1000)`, then `accuracy_score`.
