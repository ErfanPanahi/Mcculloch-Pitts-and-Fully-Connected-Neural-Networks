In this question, we will become familiar with the **AdaLine** and **MadaLine** networks.

## AdaLine Neural Network

Suppose our data in two dimensions is defined as follows: (x,y)

*	**$x$**: a normally distributed variable with mean $μ_x$ and standard deviation $σ_x$  
*	**$y$**: a normally distributed variable with mean $μ_y$ and standard deviation $σ_y$

Now we consider two groups as follows:

**Group one:** Contains 100 data points, where the variable $x$ has a mean of 0 and a standard deviation of 0.1, and the variable $y$ also has a mean of 0 and a standard deviation of 0.4.

<p align="center">
  $x \sim N(0,0.1)$ ,   $y \sim N(0,0.4)$
</p>

**Group two:** Contains 100 data points, where the variable $x$ has a mean of 1 and a standard deviation of 0.2, and the variable $y$ also has a mean of 1 and a standard deviation of 0.2.

<p align="center">
  $x \sim N(1,0.2)$ ,   $y \sim N(1,0.2)$
</p>

```python
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
```

```python
X, T = prepare_data([0, 1], [0, 1], [0.1, 0.2], [0.4, 0.2], 100)
```

The data for both groups is shown in the figure below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/25db07b6-bfc9-4b6f-871e-e358afbc64e5" width="600" height="500" >
</p>

Now, following the AdaLine learning algorithm, we first initialize the weights, bias, and learning rate with random and small values, and then update the weights and biases until the cost function decreases below a predetermined threshold. The changes in the cost function based on the samples are shown in the figure below.

```python
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
```

```python
W, b, J = train_AdaLine(X, T)
plot_cost(J)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/33f07ee9-ba1d-4b51-8c6b-b4e50258230f" width="600" height="500" >
</p>

As you can see, the cost function has decreased as learning progresses. The method of separating the two classes using the AdaLine network is shown in the figure below. ($accuracy=100$%)

```python
def prediction(X, W, b):
    h = np.zeros(np.shape(X)[1])
    for i in range(np.shape(X)[1]):
        net = np.matmul(W.T, X[:, i]) + b
        h[i] = np.sign(net)
    return h

def accuracy(X, T, W, b):
    return sum((T == prediction(X, W, b)).astype(int)) / len(T)

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
```

```python
acc = accuracy(X, T, W, b)
visual_result(X, T, W, b, acc)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/c80e6c6f-2124-4a39-b53c-79bed1eebd26" width="600" height="500" >
</p>

Now suppose we define two new data groups as follows:

**Group one:** Contains 100 data points, where the variable $x$ has a mean of 0 and a standard deviation of 0.4, and the variable $y$ also has a mean of 0 and a standard deviation of 0.4.

<p align="center">
  $x \sim N(0,0.4)$ ,   $y \sim N(0,0.4)$
</p>

**Group two:** Contains 100 data points, where the variable $x$ has a mean of 1 and a standard deviation of 0.3, and the variable $y$ also has a mean of 1 and a standard deviation of 0.3.

<p align="center">
  $x \sim N(1,0.3)$ ,   $y \sim N(1,0.3)$
</p>

```python
X, T = prepare_data([0, 1], [0, 1], [0.4, 0.3], [0.4, 0.3], 100)
```

We want to test the designed AdaLine network on the new data whose variance has increased. The result is shown in the figure below. As expected, the separation for the new data also has suitable accuracy. ($accuracy = 96$%)

```python
acc = accuracy(X, T, W, b)
visual_result(X, T, W, b, acc)
```


<p align="center">
  <img src="https://github.com/user-attachments/assets/abc7f3b0-a5a9-4794-a27d-e04bdcee7d04" width="600" height="500" >
</p>

## MAdaLine Neural Network

In this question, we aim to classify the given data [`MadaLine.csv`](https://github.com/ErfanPanahi/Mcculloch-Pitts-and-Fully-Connected-Neural-Networks/blob/main/Q2/MadaLine.csv), which consists of two classes. The scatter plot of this data is shown in the figure below.

```python
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
```

```python
X, T, X_train, X_test, T_train, T_test = read_csv_file()
visual_data(X, T)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/6167f3da-4bfd-4eda-a68d-834979f5ab17" width="600" height="500" >
</p>

The MAdaLine network design has two algorithms mentioned in the [`MAdaLine_Algorithms.pd`](https://github.com/ErfanPanahi/Mcculloch-Pitts-and-Fully-Connected-Neural-Networks/blob/main/Q2/MAdaLine_Algorithms.pdf) file. We will use the **MR-I algorithm** for designing the network related to our problem. This algorithm is shown in the figure below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4e1dcc76-6576-44fc-b5d6-98ec6281b10a" width="600" height="900" >
</p>

To design the network, we consider three cases where the number of neurons in the hidden layer (the number of AdaLine algorithm lines) is 3, 5, or 10.

We will use 90% of the data for training and keep the remaining 10% for testing the designed network at the end. 

```python
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
```

```python
ITR = 500
W3, B3, V3, b3, J3 = train_MAdaLine(X_train, T_train, 3, ITR)
acc3 = accuracy(X_test, T_test, W3, B3, V3, b3)
W5, B5, V5, b5, J5 = train_MAdaLine(X_train, T_train, 5, ITR)
acc5 = accuracy(X_test, T_test, W5, B5, V5, b5)
W10, B10, V10, b10, J10 = train_MAdaLine(X_train, T_train, 10, ITR)
acc10 = accuracy(X_test, T_test, W10, B10, V10, b10)
```

The figure below shows the accuracy and the decreasing cost function graph per iteration for each of the three mentioned cases of the number of neurons in the hidden layer.

```python
def MAdaLine_Result(J3, J5, J10, acc3, acc5, acc10):
    print('Accuracy (m = 3) = {}%'.format(acc3 * 100))
    print('Accuracy (m = 5) = {}%'.format(acc5 * 100))
    print('Accuracy (m = 10) = {}%'.format(acc10 * 100))
    plt.plot(range(len(J3)), J3, 'b', range(len(J5)), J5, 'r', range(len(J10)), J10, 'g')
    plt.legend(['3 neurons', '5 neurons', '10 neurons'])
    plt.xlabel('Iterations')
    plt.ylabel('J')
    plt.title('Cost Function for different value of m')
```

```python
MAdaLine_Result(J3, J5, J10, acc3, acc5, acc10)
```

As observed, increasing the number of neurons in the hidden layer (or, in other words, increasing the number of AdaLines and enlarging the polygon separating the two classes) results in higher network accuracy. The cost function also exhibits a more pronounced downward trend, requiring fewer iterations to reach a stable state.


<p align="center">
  <img src="https://github.com/user-attachments/assets/f49f6016-2035-43c1-98f7-36685b83291a" width="200" height="50" >
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/871b8267-4ddb-44f6-b5ba-1d49648e076e" width="600" height="500" >
</p>
