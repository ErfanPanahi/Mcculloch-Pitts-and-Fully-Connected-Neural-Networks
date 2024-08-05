import numpy as np
import random 
import matplotlib.pyplot as plt
import pandas as pd

def train_MAdaLine(X, T, m, ITR):
    V = np.ones(m); b = 0.5
    W = np.array([np.random.uniform(0, 0.1, m), np.random.uniform(0, 0.1, m)])
    B = np.random.uniform(0, 0.1, m)
    alpha = np.random.uniform(0, 0.1)
    J = np.array([0.5 * np.sum((T - prediction(X, W, B, V, b)) ** 2)])
    for _ in range(ITR):
        for i in range(len(T)):
            Z_in = np.matmul(W.T, X[:, i]) + B
            Z = np.sign(Z_in)
            Y_in = np.matmul(V.T, Z) + b
            Y = np.sign(Y_in)
            # Update Parameters
            if T[i] == Y: continue
            elif T[i] == 1:
                j = np.argmin(abs(Z_in))
                W[:, j] = W[:, j] + alpha * (1 - Z_in[j]) * X[:, i]
                B[j] = B[j] + alpha * (1 - Z_in[j]) 
            elif T[i] == -1:
                k = Z == 1
                W[:, k] = W[:, k] + alpha * np.matmul(np.array([X[:, i]]).T, np.array([-1 - Z_in[k]]))
                B[k] = B[k] + alpha * (-1 - Z_in[k])
        J = np.append(J, 0.5 * np.sum(((T - prediction(X, W, B, V, b))) ** 2)) 
    return W, B, V, b, J

def prediction(X, W, B, V, b):
    label = np.zeros(np.shape(X)[1])
    for i in range(np.shape(X)[1]):
        Z_in = np.matmul(W.T, X[:, i]) + B
        Z = np.sign(Z_in)
        Y_in = np.matmul(V.T, Z) + b
        label[i] = np.sign(Y_in)
    return label

def accuracy(X_test, T_test, W, B, V, b):
    return np.sum(prediction(X_test, W, B, V, b) == T_test) / len(T_test) 

def MAdaLine_Result(J3, J5, J10, acc3, acc5, acc10):
    print('Accuracy (m = 3) = {}%'.format(acc3 * 100))
    print('Accuracy (m = 5) = {}%'.format(acc5 * 100))
    print('Accuracy (m = 10) = {}%'.format(acc10 * 100))
    plt.plot(range(len(J3)), J3, 'b', range(len(J5)), J5, 'r', range(len(J10)), J10, 'g')
    plt.legend(['3 neurons', '5 neurons', '10 neurons'])
    plt.xlabel('Iterations')
    plt.ylabel('J')
    plt.title('Cost Function for different value of m')

def read_csv_file():
    df = pd.read_csv("MAdaLine.csv")
    x = np.append(np.array(df[df.columns[0]].to_list()), float(df.columns[0]))
    y = np.append(np.array(df[df.columns[1]].to_list()), float(df.columns[1]))
    shuffle_ind = np.arange(len(x))
    random.shuffle(shuffle_ind)
    X = np.array([x, y])[:, shuffle_ind]
    T = np.append(np.array(df[df.columns[2]].to_list()), float(df.columns[2]))[shuffle_ind]
    T[T == 0] = -1
    X_train = X[:, :180]; T_train = T[:180]
    X_test = X[:, 180:]; T_test = T[180:]
    return X, T, X_train, X_test, T_train, T_test

def visual_data(X, T):
    x1 = []; x2 = []; y1 = []; y2 = []
    for i in range(len(T)):
        if T[i] == -1: 
            x1.append(X[0, i])
            y1.append(X[1, i])
        else:
            x2.append(X[0, i])
            y2.append(X[1, i])
    plt.plot(x1, y1,'go', x2, y2,'bo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot')


