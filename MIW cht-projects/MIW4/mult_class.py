import numpy as np

class MultiClassClassifier:
    def __init__(self, p1, p2) -> None:
        self.p1 = p1
        self.p2 = p2

    def predict(self, X):
        # print(self.p1.predict(X))
        # print(self.p2.predict(X))
        return np.where(self.p1.predict(X) == 1, 0, 
                np.where(self.p2.predict(X) == 1, 2, 1))
