NOTES:<br>
Attribute Information: <br>
1.model - > Ford Car Brands<br>
2.year - >Production Year <br>
3.price - >Price of car in $ <br>
4.transmission - > Automatic,Manual, Semi-Auto <br>
5.mileage -> Number of miles traveled <br>
6.fuel_Type -> Petrol,Diesel,Hybrid,Electric,Other <br>
7.tax -> Annual Tax <br>
8.mpg - > Miles per Gallon <br>
9.engineSize - > Car's Engine Size <br>

# Ford used-car price — linear regression

Predict `price` from Ford listings in `ford.csv`. Pipeline: EDA → encode categories two ways → scale → linear regression.

## EDA

`project.py` plots:

- price histogram (with KDE)
- numeric correlation heatmap
- price vs `year`, `mileage`, `engineSize`, `transmission`, `fuelType`, `model`

## Features and target

```python
X = df.drop(columns=['price'])
y = df['price']
```

Two encodings of the same `X`:

| Version | What it does |
| --- | --- |
| `X_one_encode` | One-hot (`get_dummies`) on `model`, `transmission`, `fuelType`. Dummy columns are 0/1; no fake order among categories. |
| `Xlabel` | `LabelEncoder` maps each category to 0, 1, 2, … Same number of columns as the original table. |

Numeric columns are then scaled with `StandardScaler` (mean 0, std 1). For one-hot, only `year`, `mileage`, `tax`, `mpg`, `engineSize` are scaled. For `Xlabel`, the encoded category columns are scaled too.

## Linear regression

80/20 train-test split (`random_state=42`). Same `LinearRegression` on both feature tables.

Typical test scores from this script:

| Encoding | R² | MAE | RMSE |
| --- | --- | --- | --- |
| One-hot | ~0.85 | ~1371 | ~1855 |
| Label | ~0.74 | ~1777 | ~2430 |

One-hot is the better fit here. Linear regression treats a number as a quantity, so label-encoded `model` (Fiesta=5, Focus=6, …) is a weaker signal than a dummy column per model.

Scatter plots of actual vs predicted price: `one_hot_linear_regression.png`, `xlabel_linear_regression.png`.

## Run

```bash
python project.py
```

Needs pandas, numpy, seaborn, matplotlib, and scikit-learn. Plots are written next to `project.py`.
