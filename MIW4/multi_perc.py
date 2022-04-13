from given.perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from given.plotka import plot_decision_regions

class MultiClassPerceptron:
    def __init__(self, p1, p2) -> None:
        self.p1 = p1
        self.p2 = p2

    def predict(self, X):
        # print(self.p1.predict(X))
        # print(self.p2.predict(X))
        return np.where(self.p1.predict(X) == 1, 0, 
                np.where(self.p2.predict(X) == 1, 2, 1))


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y_train_0 = y_train.copy()
    y_train_0[y_train_0 != 0] = -1
    y_train_0[y_train_0 == 0] = 1

    y_train_1 = y_train.copy()
    y_train_1[y_train_1 != 2] = -1
    y_train_1[y_train_1 == 2] = 1

    print(y_train_0)
    print(y_train_1)

    p1 = Perceptron(n_iter=250)
    p1.fit(X_train, y_train_0)
    p2 = Perceptron(n_iter=250)
    p2.fit(X_train, y_train_1)

    mcp = MultiClassPerceptron(p1, p2)
    plot_decision_regions(X=X_train, y=y_train, classifier=mcp)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()
