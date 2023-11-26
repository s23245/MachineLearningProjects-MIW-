import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./Dane/dane7.txt', header=None, sep='\s', engine='python')

X = data.iloc[:, [0]].values
y = data.iloc[:, [1]].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# y = ax + b
cost = np.hstack([X_train, np.ones(X_train.shape)])
v = np.linalg.pinv(cost) @ y_train

print('model 1: y=ax+b')
print(f'v:\n{v}')

plt.plot(X_test, y_test, 'ro')
plt.plot(X_test, v[0]*X_test + v[1], 'b*')
plt.show()

# polynomial (9)
cost = np.hstack([
    X_train ** 9,
    X_train ** 8,
    X_train ** 7,
    X_train ** 6,
    X_train ** 5,
    X_train ** 4,
    X_train ** 3,
    X_train ** 2,
    X_train,
    np.ones(X_train.shape)
])
v = np.linalg.pinv(cost) @ y_train

print('model 2: y=ax2 + bx + c')
print(f'v:\n{v}')

plt.plot(X_test, y_test, 'ro')
plt.plot(X_test, 
        v[0] * (X_test ** 9) +
        v[1] * (X_test ** 8) +
        v[2] * (X_test ** 7) +
        v[3] * (X_test ** 6) +
        v[4] * (X_test ** 5) +
        v[5] * (X_test ** 4) +
        v[6] * (X_test ** 3) +
        v[7] * (X_test ** 2) +
        v[8] * X_test + 
        v[9], 'b*')
plt.show()

