In this question, we have a dataset for predicting prices [`CarPrice_Assignment.csv`](https://github.com/ErfanPanahi/Mcculloch-Pitts-and-Fully-Connected-Neural-Networks/blob/main/Q4/CarPrice_Assignment.csv). First, we will work with the data and get familiar with ***feature engineering***. Then, we will use a **Multi-Layer Perceptron (MLP)** network to predict prices and compare the predictions with the actual prices.

The aim of this question is to become familiar with Multi-Layer Perceptron (MLP) and the `TensorFlow` and `keras` libraries.

First, we will read the data using `pandas` package.

```python
Data = pd.read_csv('CarPrice_Assignment.csv')
```

The figure below shows the total number of NaN values in each feature.

```python
Data.isna().sum()
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/7c5ff703-74e1-4117-833a-e5f1e82d3c44" width="200" height="600" >
</p>

## Introduction to the Dataset and Preprocessing

In this section, we will preprocess the data to prepare it for network training. The preprocessing steps are as follows:
1.	**Extract Company Name:** Separate the company name from the `CarName` column and store it in a new column named `CompanyName`. (I will also correct any incorrectly entered company names.)
2.	**Drop Columns:** Remove the columns `CarName`, `car_ID`, `and symbolling`.
3.	**Convert Categorical Data:** Convert descriptive data to numeric data using `pd.get_dummies()`.


```python
def preprocessing(Data):
    CompanyName = np.array([])
    for i in range(len(Data.CarName)):
        k = Data.CarName[i].find(' ')
        CompanyName = np.append(CompanyName, Data.CarName[i][:k])

    CompanyName[CompanyName == 'maxda'] = 'mazda'
    CompanyName[CompanyName == 'toyouta'] = 'toyota'
    CompanyName[CompanyName == 'subar'] = 'subaru'
    CompanyName[CompanyName == 'vokswagen'] = 'volkswagen'
    CompanyName[CompanyName == 'vw'] = 'volkswagen'


    Data = Data.drop(['CarName', 'car_ID', 'symboling'], axis = 1)
    Data.insert(0, 'CompanyName', CompanyName)

    preprocessed_Data = pd.get_dummies(Data)

    return preprocessed_Data
```

```python
preprocessed_Data = preprocessing(Data)
```

Next, we use the correlation matrix to identify the feature that has the highest correlation with the price (target). 

```python
def max_corr(Data):
    correlation = pd.DataFrame.corr(Data)
    k = np.argmax(correlation.price.drop('price'))
    feature = correlation.columns[k]
    plt.scatter(Data[feature], Data.price)
    plt.xlabel(feature)
    plt.ylabel('price')
    plt.title('Maximum Correlated Feature with Price')
```

```python
max_corr(preprocessed_Data)
```

The figure below shows the distribution of price and enginesize (which has the highest correlation with the price).


<p align="center">
  <img src="https://github.com/user-attachments/assets/4fd88070-3dab-4e1a-8171-9ede4684eb10" width="400" height="300" >
</p>

Finally, we will split the data into two parts: 85% for training and 15% for testing. This will prepare the data for training the network.

```python
def Extract_Train_Test(Data, split_per):
    t = np.where(Data.columns == 'price')
    Data_final = Data.values
    np.random.shuffle(Data_final)
    T = Data_final[:, t].T[0][0]
    X = np.delete(Data_final, t, axis = 1)

    split = int(len(T) * split_per)
    X_train = X[:split]
    X_test = X[split:]
    T_train = T[:split]
    T_test = T[split:]

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.expand_dims(np.asarray(X_train).astype(np.float32), -1)
    X_test = np.expand_dims(np.asarray(X_test).astype(np.float32), -1)
    T_train = np.expand_dims(np.asarray(T_train).astype(np.float32), -1)
    T_test = np.expand_dims(np.asarray(T_test).astype(np.float32), -1)

    return X_train, T_train, X_test, T_test
```

```python
X_train, T_train, X_test, T_test = Extract_Train_Test(preprocessed_Data, 0.85)
```

## Multi-Layer Perceptron 

In this section, we will first design a network with 3 hidden layers for predicting car prices.

```python
def network(layers_len):
    inputs = Input(shape = layers_len[0])
    net = inputs
    net = Flatten()(net)
    net = Dense(layers_len[1], activation='relu')(net)
    net = Dense(layers_len[2], activation='relu')(net)
    net = Dense(layers_len[3], activation='relu')(net)
    net = Dense(layers_len[4], kernel_initializer='normal', activation='linear')(net)
    mlp = Model(inputs, net)
    return mlp
```

```python
layers_len = [X_train[0].shape, 200, 150, 100, 1]
mlp = network(layers_len)
```

The figure below shows the architecture of the network.

<p align="center">
  <img src="https://github.com/user-attachments/assets/118628a8-8c89-436e-85dc-669b9d076e0c" width="400" height="150" >
</p>

I will train the network using the `MeanAbsolutePercentageError` loss function and the `adamw` optimizer. 

```python
def training(model, Epoch_Num, X_train, T_train, X_test, T_test):
    model.compile(optimizer='adamw', loss='mape', metrics=['r2_score'])
    History = model.fit(X_train, T_train, 
                        epochs = Epoch_Num,
                        batch_size = 10,
                        validation_data = (X_test, T_test))
    return History
```

```python
History = training(mlp, 250, X_train, T_train, X_test, T_test)
```


The figure below shows the loss function and R2-score metrics for the training and test datasets over the course of training (as a function of epochs).

<p align="center">
  <img src="https://github.com/user-attachments/assets/eba547ee-1dee-455c-b926-b555d7f8bea7" width="800" height="650" >
</p>


The figure below shows the scatter plot of predicted prices versus actual prices. As observed, the car prices are predicted with good accuracy, achieving an error of approximately 10%.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3a213ae9-34b1-4939-96ad-e5b7483abc6c" width="800" height="380" >
</p>


The comparison of predicted prices with actual prices for 5 random cars from the test dataset is shown in the figure below. As observed, the predicted prices are quite close to the actual prices.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5ba1e846-8f00-435d-b49b-8e4a1a1cdc64" width="300" height="300" >
</p>


