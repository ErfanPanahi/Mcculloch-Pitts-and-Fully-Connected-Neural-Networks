import numpy as np
import random 
import matplotlib.pyplot as plt

def prediction(X, W, b):
    h = np.zeros(np.shape(X)[1])
    for i in range(np.shape(X)[1]):
        net = np.matmul(W.T, X[:, i]) + b
        h[i] = np.sign(net)
    return h

def train_AdaLine(X, T):
    W = np.random.uniform(0, 0.1, 2)
    b = np.random.uniform(0, 0.1)
    alpha = np.random.uniform(0, 0.1)
    J = []
    for i in range(len(T)):
        net = np.matmul(W, X[:, i]) + b
        error = T[i] - net
        J.append(0.5 * error**2)
        if J[i] < 1e-5:
            break
        W = (W + alpha * error * X[:, i])
        b = b + alpha * error
    return W, b, J

def plot_cost(J):
    plt.plot(J)
    plt.xlabel('sample (i)')
    plt.ylabel('Cost Function (J)')
    plt.title('Cost Function - AdaLine Network')

def visual_result(X, T, W, b, acc):
    x1 = []; x2 = []; y1 = []; y2 = []
    for i in range(len(T)):
        if T[i] == -1: 
            x1.append(X[0, i])
            y1.append(X[1, i])
        else:
            x2.append(X[0, i])
            y2.append(X[1, i])
    x = np.array([np.min([[x1,x2]]), np.max([x1,x2])])
    y = -W[0] / W[1] * x - b / W[1]
    print('Accuracy: {}%'.format(acc * 100))
    plt.plot(x1, y1,'go', x2, y2,'bo', x, y,'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot - AdaLine Network')


def accuracy(X, T, W, b):
    return sum((T == prediction(X, W, b)).astype(int)) / len(T)

def prepare_data(mu_x, mu_y, sigma_x, sigma_y, n):
    x1 = np.random.normal(mu_x[0], sigma_x[0], n)
    y1 = np.random.normal(mu_y[0], sigma_y[0], n)
    x2 = np.random.normal(mu_x[1], sigma_x[1], n)
    y2 = np.random.normal(mu_y[1], sigma_y[1], n)
    shuffle_ind = np.arange(len(x1) + len(x2))
    random.shuffle(shuffle_ind)
    X = np.array([np.append(x1, x2), np.append(y1, y2)])[:, shuffle_ind]
    T = np.append(-np.ones(len(x1)), np.ones(len(x1)))[shuffle_ind]
    return X, T
    
# x1 = np.random.normal(0, 0.1, 100)
# y1 = np.random.normal(0, 0.4, 100)
# x2 = np.random.normal(1, 0.2, 100)
# y2 = np.random.normal(1, 0.2, 100)

# shuffle_ind = np.arange(len(x1) + len(x2))
# random.shuffle(shuffle_ind)
# X = np.array([np.append(x1, x2), np.append(y1, y2)])[:, shuffle_ind]
# T = np.append(-np.ones(len(x1)), np.ones(len(x1)))[shuffle_ind]

# W, b = train(X, T)

# acc = accuracy(X, T, W, b)
# visual_result(x1, x2, y1, y2, W, b, acc)