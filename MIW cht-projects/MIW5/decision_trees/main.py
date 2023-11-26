import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from plotka import plot_decision_regions

seed = 228  # to avoid random state to change the output

def classify_and_plot(X_train, y_train, X_test, y_test, classifier, str_args, filename):
    classifier.fit(X_train, y_train)
    plot_decision_regions(
        np.vstack((X_train, X_test)),
        np.hstack((y_train, y_test)),
        classifier=classifier,
        test_idx=range(105, 150)
    )
    plt.title(f'{classifier.__class__.__name__} {str_args}')
    plt.xlabel('Petel length')
    plt.ylabel('Petel width')
    plt.savefig(filename)
    plt.close()
    # plt.show()
    

if __name__ == '__main__':

    # preprocessing

    data = load_iris()
    X = data.data[:, [2, 3]]
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    sc = StandardScaler()
    sc.fit(X_train)
    tr_X_train = sc.transform(X_train)
    tr_X_test = sc.transform(X_test)

    # decision trees gini and entropy max=3
    max_depth = 3
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=seed)
    classify_and_plot(
        X_train, y_train,
        X_test, y_test,
        classifier,
        f'gini-{max_depth}',
        f'tree-gini-{max_depth}'
    )

    classifier = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=seed)
    classify_and_plot(
        X_train, y_train,
        X_test, y_test,
        classifier,
        f'entropy-{max_depth}',
        f'tree-entropy-{max_depth}'
    )

    # decision trees gini max_depth = 4 & 8

    max_depth = 4
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=seed)
    classify_and_plot(
        X_train, y_train,
        X_test, y_test,
        classifier,
        f'gini-{max_depth}',
        f'tree-gini-{max_depth}'
    )

    max_depth = 8
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=seed)
    classify_and_plot(
        X_train, y_train,
        X_test, y_test,
        classifier,
        f'gini-{max_depth}',
        f'tree-gini-{max_depth}'
    )

    # random forest trees = 3 & 9

    n_est = 3
    classifier = RandomForestClassifier(criterion='gini', n_estimators=n_est, random_state=seed)
    classify_and_plot(
        X_train, y_train,
        X_test, y_test,
        classifier,
        f'gini-{n_est}',
        f'forest-gini-{n_est}'
    )

    n_est = 9
    classifier = RandomForestClassifier(criterion='gini', n_estimators=n_est, random_state=seed)
    classify_and_plot(
        X_train, y_train,
        X_test, y_test,
        classifier,
        f'gini-{n_est}',
        f'forest-gini-{n_est}'
    )

