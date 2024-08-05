import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

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

def max_corr(Data):
    correlation = pd.DataFrame.corr(Data)
    k = np.argmax(correlation.price.drop('price'))
    feature = correlation.columns[k]
    plt.scatter(Data[feature], Data.price)
    plt.xlabel(feature)
    plt.ylabel('price')
    plt.title('Maximum Correlated Feature with Price')

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

def training(model, Epoch_Num, X_train, T_train, X_test, T_test):
    model.compile(optimizer='adamw', loss='mape', metrics=['r2_score'])
    History = model.fit(X_train, T_train, 
                        epochs = Epoch_Num,
                        batch_size = 10,
                        validation_data = (X_test, T_test))
    return History

def training_results(History):
    plt.figure(figsize = (12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(History.history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training Data')
    plt.subplot(2, 2, 2)
    plt.plot(History.history['val_loss'], 'g')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Validation Data')
    plt.subplot(2, 2, 3)
    plt.plot(History.history['r2_score'])
    plt.xlabel('epoch')
    plt.ylabel('R2 score')
    plt.subplot(2, 2, 4)
    plt.plot(History.history['val_r2_score'], 'g')
    plt.xlabel('epoch')
    plt.ylabel('R2 score')
    plt.suptitle('Training Results (Loss and R2-Score)')

def visual_results(model, X_train, T_train, X_test, T_test, n):
    T_train_pred = model.predict(X_train)
    T_test_pred = model.predict(X_test)
    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(T_train, T_train_pred)
    plt.plot([min(T_train), max(T_train)], [min(T_train), max(T_train)], 'r')
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.title('Train')
    plt.subplot(1, 2, 2)
    plt.scatter(T_test, T_test_pred)
    plt.plot([min(T_test), max(T_test)], [min(T_test), max(T_test)], 'r')
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.title('Test')
    plt.suptitle('Result (Price) Comparison')
    for i in range(n):
        rnd = np.random.randint(0, len(T_test))
        print('\nReal Price for Car Number {}: {}'.format(rnd, T_test[rnd]))
        print('Predicted Price for Car Number {}: {} \n'.format(rnd, T_test_pred[rnd]))
