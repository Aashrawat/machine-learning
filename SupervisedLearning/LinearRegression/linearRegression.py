import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("Housing.csv")
Y = df['price']
X = df['lotsize']
X = X.to_numpy().reshape(len(X),1)
Y = Y.to_numpy().reshape(len(Y),1)
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2, random_state=42)
plt.scatter(X_test, Y_test, color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())
regr =linear_model.LinearRegression()
regr.fit(X_train, Y_train)
plt.plot (X_test, regr.predict(X_test), linewidth = 3, color='red')
plt.savefig("regression_plot.png")
print("Plot saved as regression_plot.png")
