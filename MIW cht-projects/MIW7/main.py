import numpy as np
import matplotlib.pyplot as plt

P = np.vstack(np.arange(-2, 2.01, 0.1)).T
T = np.power(P, 2) + 1 * (np.random.rand(P.shape[0], P.shape[1]) - 0.5)

# network
S1 = 100
W1 = np.random.rand(S1, 1) - 0.5
B1 = np.random.rand(S1, 1) - 0.5
W2 = np.random.rand(1, S1) - 0.5
B2 = np.random.rand(1, 1) - 0.5
lr = 0.001

for i in range(1, 301):
    s = W1 @ P + B1 @ np.ones(P.shape)
    A1 = np.arctan(s) # activation sigmoid
    A1 = A1.clip(min=0) # activation relu
    A2 = W2 @ A1 + B2 

    E2 = T - A2
    E1 = W2.T @ E2

    dW2 = lr * E2 @ A1.T
    dB2 = lr * E2 @ np.ones(E2.shape).T
    dW1 = lr * (1 / (1 + np.power(s, 2))) * E1 @ P.T
    dB1 = lr * (1 / (1 + np.power(s, 2))) * E1 @ np.ones(P.shape).T

    W2 = W2 + dW2
    B2 = B2 + dB2
    W1 = W1 + dW1
    B1 = B1 + dB1

    if i % 100 == 0:
        plt.plot(P, T, 'r*')
        plt.plot(P, A2, 'go')

plt.show()
