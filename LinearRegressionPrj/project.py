import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("ford.csv")

print(df.head())

print(df.shape)

print(df.info())

print(df.describe())

print(df.isnull().sum())

#EDA 
sns.histplot(df['price'], bins = 50, kde=True)
plt.savefig('price_hist.png')
plt.close()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.savefig('correlation_heatmap.png')
plt.close()
sns.boxplot(data=df, x='year', y='price')
plt.xticks(rotation=90)
plt.savefig('year_price_boxplot.png')
plt.close()
sns.scatterplot(data=df, x='mileage', y ='price')
plt.savefig('mileage_price_scatterplot.png')
plt.close()
sns.boxplot(data=df, x='engineSize', y='price')
plt.savefig('engineSize_price_boxplot.png')
plt.close()
sns.boxplot(data=df, x='transmission', y='price')
plt.savefig('transmission_price_boxplot.png')
plt.close()
sns.boxplot(data=df, x='fuelType', y='price')
plt.savefig('fuelType_price_boxplot.png')
plt.close()
sns.boxplot(x=df['model'], y =df['price'])
plt.xticks(rotation=90)
plt.savefig('model_price_boxplot.png')
plt.close()

X = df.drop(columns=['price'])  # price we want it in output
y = df['price']
print(X.head())
print(y.head())

X_one_encode = pd.get_dummies(X, columns=['model', 'transmission', 'fuelType'], drop_first=True)
X_one_encode = X_one_encode.astype(int)  # one hot encoding converts categorical data into numbers so the model can use them
print(X_one_encode.head())


from sklearn.preprocessing import LabelEncoder

columns = ['model', 'transmission', 'fuelType']

Xlabel = X.copy()

label_encoders = {}

for col in columns:
    le = LabelEncoder()
    Xlabel[col] = le.fit_transform(Xlabel[col].astype(str))
    label_encoders[col] = le

print(Xlabel.head())

from sklearn.preprocessing import StandardScaler
numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
scaler = StandardScaler()
X_one_encode[numerical_cols] = scaler.fit_transform(X_one_encode[numerical_cols])
print(X_one_encode.head())

Xlabel[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg',
       'engineSize']] = scaler.fit_transform(Xlabel[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg',
       'engineSize']])
print(Xlabel.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

datasets = {
    'one_hot': X_one_encode,
    'xlabel': Xlabel,
}

for name, X_features in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(name)
    print('R2:', r2_score(y_test, y_pred))
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('Actual price')
    plt.ylabel('Predicted price')
    plt.title(f'Linear Regression ({name})')
    plt.savefig(f'{name}_linear_regression.png')
    plt.close()

