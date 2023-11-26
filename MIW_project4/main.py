import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('./Dane/dane1.txt', header=None, sep='\s', engine='python')

X = data.iloc[:, [0]].values
y = data.iloc[:, [1]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# MODEL 1 y = ax + b
cost = np.hstack([X_train, np.ones(X_train.shape)])
v = np.linalg.pinv(cost) @ y_train

print('model 1: y=ax+b')
print(f'v:\n{v}')

plt.plot(X_test, y_test, 'ro')
plt.plot(X_test, v[0] * X_test + v[1], 'b*')
plt.show()

mse_model1 = mean_squared_error(y_test, v[0] * X_test + v[1])
mse_model1_train = mean_squared_error(y_train, v[0] * X_train + v[1])
# MODEL 2 y = ax^2 + bx + c

cost = np.hstack([
    X_train ** 2,
    X_train,
    np.ones(X_train.shape)
])
v = np.linalg.pinv(cost) @ y_train

print('model 2: y=ax^2 + bx + c')
print(f'v:\n{v}')

plt.plot(X_test, y_test, 'ro')
plt.plot(X_test,
         v[0] * (X_test ** 2) +
         v[1] * X_test +
         v[2], 'b*')
plt.show()

mse_model2 = mean_squared_error(y_test, v[0] * X_test ** 2 + v[1] * X_test + v[2])
mse_model2_train = mean_squared_error(y_train, v[0] * X_train ** 2 + v[1] * X_train + v[2])

print("MSE(train) MODEL 1: ", mse_model1_train)
print("MSE(train) MODEL 2: ", mse_model2_train)

print("MSE(test) MODEL 1: ", mse_model1)
print("MSE(test) MODEL 2: ", mse_model2)
