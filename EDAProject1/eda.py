import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('insurance.csv')

#EDA

print (df.shape)

print (df.head())

print (df.info())

print (df.describe())

print (df.isnull().sum())

print (df.columns)

numeric_columns =['age', 'bmi', 'charges']
for column in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True, bins=20) ## kde = kernel density estimation
    plt.savefig(f"{column}_hist.png")
    plt.close()
    plt.figure(figsize=(6, 4))
    sns.countplot(x='children', data=df)
    plt.savefig('children_count.png')
    plt.close()
    sns.countplot(x=df['sex'])
    plt.savefig('sex_count.png')
    plt.close()
    sns.countplot(x=df['smoker'])
    plt.savefig('smoker_count.png')
    plt.close()
    sns.countplot(x=df['region'])
    plt.savefig('region_count.png')
    plt.close()
for column in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[column])
    plt.savefig(f"{column}_boxplot.png")
    plt.close()
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)## annot = True means show the correlation values on the heatmap
plt.savefig('correlation_heatmap.png')
plt.close()

##DATA CLEANING AND PREPROCESSING

df_cleaned=df.copy()
print(df_cleaned.head())

print(df_cleaned.info())

print(df_cleaned.isnull().sum())

print(df_cleaned.describe())

print(df_cleaned.columns)

print(df_cleaned.shape)

print(df_cleaned.head())

print(df_cleaned.info())

print(df['region'].value_counts())

df_cleaned = pd.get_dummies(df_cleaned,columns = ['region'],drop_first=True)

print(df_cleaned.head())

sns.histplot(df['bmi'])
plt.savefig('bmi_hist.png')
plt.close()

df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)
print(df_cleaned)

df_cleaned['is_female'] = df_cleaned['sex'].map({'male': 0, 'female': 1})
df_cleaned['is_smoker'] = df_cleaned['smoker'].map({'no': 0, 'yes': 1})
df_cleaned = df_cleaned.drop(columns=['sex', 'smoker'])

df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
df_cleaned = df_cleaned.astype(int)
print(df_cleaned.head())
print(df_cleaned.columns)

from sklearn.preprocessing import StandardScaler
cols = ['age', 'bmi', 'children']
scaler = StandardScaler()

df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head())

from scipy.stats import pearsonr, chi2_contingency

selected_features = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}
correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation'])
correlation_df = correlation_df.sort_values(by='Pearson Correlation', ascending=False)
print(correlation_df)

cat_features = [
    'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)
chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')
print(chi2_df)

final_df = df_cleaned[['age', 'is_female', 'bmi', 'children', 'is_smoker', 'charges', 'region_southeast', 'bmi_category_Obese']]
print(final_df)
print(df)

