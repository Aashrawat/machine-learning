# EDA, Data Cleaning, Preprocessing, and Feature Engineering

This folder is an end-to-end pass over medical insurance charges (`insurance.csv`): explore the raw table, clean and encode it, engineer new features, then keep only the ones that actually relate to `charges`.

## Exploratory data analysis (EDA)

EDA is looking at the data **before** modeling so you know shape, types, missing values, and distributions.

In `eda.py` that means:

- `df.shape`, `df.head()`, `df.info()`, `df.describe()`, `df.isnull().sum()`
- histograms with **KDE** (kernel density estimation) for `age`, `bmi`, `charges`
- count plots for `children`, `sex`, `smoker`, `region`
- box plots for outliers
- a correlation heatmap (`annot=True` writes the numbers on each cell)

## Data cleaning and preprocessing

Cleaning is making the table usable. This dataset has no missing values, so the work is encoding and scaling.

| Step | What it does |
| --- | --- |
| `df.copy()` | Work on `df_cleaned` so the original `df` stays intact. |
| One-hot encode `region` | `pd.get_dummies(..., drop_first=True)` turns categories into 0/1 columns and drops one region to avoid dummy-variable trap. |
| Map `sex` / `smoker` | `is_female` and `is_smoker` are binary flags. |
| `astype(int)` | Booleans become `0` / `1`. |
| `StandardScaler` | `age`, `bmi`, and `children` are scaled to mean $0$ and std $1$ so they sit on the same numeric scale. |

`charges` is left unscaled because it is the **target**.

## Feature engineering and extraction

**Feature engineering** is creating new columns from existing ones. **Feature extraction / selection** is deciding which of those columns to keep.

Engineering:

- `bmi_category` from WHO-style BMI bins: Underweight / Normal / Overweight / Obese (`pd.cut`)
- dummy columns from that category (`bmi_category_Normal`, `Overweight`, `Obese`; Underweight dropped)

Extraction (which features matter for `charges`):

- **Pearson correlation** — linear association with `charges`. `is_smoker` dominates (~$0.79$), then `age` and obese BMI.
- **Chi-square** — independence test of each categorical flag vs quartile-binned charges. Keep a feature if $p < 0.05$ (reject the null of “no association”).

`final_df` keeps:

`age`, `is_female`, `bmi`, `children`, `is_smoker`, `charges`, `region_southeast`, `bmi_category_Obese`

## Run

```bash
python eda.py
```

Needs pandas, numpy, seaborn, matplotlib, scikit-learn, and scipy.
