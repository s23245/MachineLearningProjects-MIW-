from given.reglog import LogisticRegressionGD
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from given.plotka import plot_decision_regions
from mult_class import MultiClassClassifier


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y_train0 = y_train.copy()
    y_train1 = y_train.copy()
    
    y_train0[y_train == 0] = 1
    y_train0[y_train != 0] = 0
    
    y_train1[y_train == 2] = 1
    y_train1[y_train != 2] = 0

    lr1 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lr2 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)

    lr1.fit(X_train, y_train0)
    lr2.fit(X_train, y_train1)

    csf = MultiClassClassifier(lr1, lr2) 

    plot_decision_regions(X=X_train, y=y_train, classifier=csf)
    plt.show()
