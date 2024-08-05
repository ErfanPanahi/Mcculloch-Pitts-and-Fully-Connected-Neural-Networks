In this question, we want to solve a classification problem using an **Auto-Encoder**. For a better understanding of Auto-Encoders, it is recommended to read the attached paper [`liu2017.pdf`](https://github.com/ErfanPanahi/Mcculloch-Pitts-and-Fully-Connected-Neural-Networks/blob/main/Q3/liu2017.pdf). The goal of this exercise is to familiarize yourself with the `keras` package and work with the `MNIST` dataset.

## Introduction to MNIST Dataset

In this section, the goal is to get familiar with and work with the dataset. You can add the dataset using the `torchvision` package as shown below. Alternatively, you can use the attached file [`mnist.npz`](https://github.com/ErfanPanahi/Mcculloch-Pitts-and-Fully-Connected-Neural-Networks/blob/main/Q3/mnist.npz).

```python
import torchvision
torchvision.datasets.MNIST (...)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

```python
def load_data():
    path = 'C:/Users/ASUS/Desktop/Deep Learning/Homeworks/HW1/mnist.npz'

    with np.load(path, allow_pickle=True) as f:
        train_images, train_labels = f['x_train'], f['y_train']
        test_images, test_labels = f['x_test'], f['y_test']

    return train_images, train_labels, test_images, test_labels
```

```python
train_images, train_labels, test_images, test_labels = load_data()
train_images = train_images.astype('float32') / np.max(train_images)
test_images = test_images.astype('float32') / np.max(test_images)
train_images_flat = train_images.reshape(len(train_images), np.prod(train_images.shape[1:]))
test_images_flat = test_images.reshape(len(test_images), np.prod(test_images.shape[1:]))
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)
```

The figure below shows 10 random images from the dataset along with their labels.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3f21915e-8408-49a0-8db2-2b3fcc47893c" width="1000" height="450" >
</p>

The figure below shows the number of data points for each label in the training dataset.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ab73707-07f7-4129-be43-8db51b6da730" width="500" height="400" >
</p>

## Auto-Encoder Network

In this section, we want to design the desired network. For this purpose, we will design two separate parts for the network: 

1.	the encoder 
2.	the decoder.

```python
def AutoEncoder(layers_shape):
    inputs = Input(shape = (layers_shape[0], ))
    enc = inputs
    enc = Flatten()(enc)
    enc = Dense(layers_shape[1], activation='relu')(enc)
    enc = Dense(layers_shape[2], activation='relu')(enc)
    enc = Dense(layers_shape[3], activation='relu')(enc)
    enc = Dense(layers_shape[4], activation='relu')(enc)
    encoder = Model(inputs, enc, name = 'encoder')
    dec_inputs = Input(shape = (layers_shape[4], ))
    dec = dec_inputs
    dec = Dense(layers_shape[3], activation='relu')(dec)
    dec = Dense(layers_shape[2], activation='relu')(dec)
    dec = Dense(layers_shape[1], activation='relu')(dec)
    dec = Dense(layers_shape[0], activation='sigmoid')(dec)
    decoder = Model(dec_inputs, dec, name = 'decoder')
    autoencoder = Model(inputs, decoder(encoder(inputs)), name = 'autoencoder')
    return autoencoder, encoder, decoder, inputs
```

```python
layers_shape_autoencoder = [784, 500, 300, 100, 30, 784]
autoencoder, encoder, decoder, inputs = AutoEncoder(layers_shape_autoencoder)
```

The figure below shows the architecture of both parts.


<p align="center">
  <img src="https://github.com/user-attachments/assets/42db361a-2f04-420b-beda-6b5b66852578" width="500" height="600" >
</p>

Now, we will train the network. For this purpose, we will use the `adam` optimizer and the `binary_crossentropy` loss function. 

```python
def train_net(model, train_data, train_labels, test_data, test_labels, Epoch_num):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    History = model.fit(train_data, train_labels, 
                             epochs = Epoch_num,
                             batch_size = 256,
                             validation_data = (test_data, test_labels))
    return History
```

```python
History_autoencoder = train_net(autoencoder, train_images_flat, train_images_flat, test_images_flat, test_images_flat, 50)
```

The figure below shows the loss function plot against the number of epochs. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4da049f-592b-4621-aa79-558987412f4b" width="850" height=350" >
</p>

The figure below compares the original images with the decoded ones (the network's output).

```python
decoded_images = autoencoder.predict(test_images_flat)
decoded_images = decoded_images.reshape(test_images.shape)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/10cef668-b49f-4822-9822-59ce7d45bfa5" width="1000" height="450" >
</p>

## Classification

In this section, I want to use a 30-dimensional feature space (the output of the encoder) to build a simple classifier with two hidden layers. To do this, I will separate the encoder part after training the autoencoder and use its outputs to train the classifier.

```python
encoded_train_images = encoder.predict(train_images_flat)
encoded_test_images = encoder.predict(test_images_flat)
```

The figure below shows the architecture of the classifier network.

```python
def mlp_2h_layer(layers_shape):
    clf_input = Input(shape = (layers_shape[0], ))
    clf = clf_input
    clf = Dense(layers_shape[1], activation='relu')(clf)
    clf = Dense(layers_shape[2], activation='relu')(clf)
    clf = Dense(layers_shape[3], activation='sigmoid')(clf)
    classifier = Model(clf_input, clf, name = 'classifier')
    return classifier
```

```python
layers_shape_classifier = [layers_shape_autoencoder[4], 60, 30, 10]
classifier = mlp_2h_layer(layers_shape_classifier)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/e0d4518a-fa6d-4783-9b6b-374d44955875" width="500" height="150" >
</p>

Now, I will train the classifier network. The figures below show the loss plot and the accuracy plot against the number of epochs for both the training and validation datasets.

```python
History_classifier = train_net(classifier, encoded_train_images, train_labels_cat, encoded_test_images, test_labels_cat, 100)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/a3974e06-2f89-442d-9782-a3afb82644ce" width="850" height="350" >
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/cdc8e674-a027-4a00-9f96-10215d1391bd" width="850" height="350" >
</p>

Finally, we connect the encoder network to the classifier network. 

```python
encoder_classifier = Model(inputs, classifier(encoder(inputs)), name = 'encoder_classifier')
```

The figure below shows the architecture of the final network along with the accuracy on the training and test datasets.

```python
Train_pred = np.argmax(encoder_classifier.predict(train_images_flat), 1)
Test_pred = np.argmax(encoder_classifier.predict(test_images_flat), 1)
train_acc = sum(Train_pred == train_labels) / len(train_labels)
test_acc = sum(Test_pred == test_labels) / len(test_labels)
print('Train Accuracy: {}%'.format(round(train_acc * 100, 2)))
print('Test Accuracy: {}%'.format(round(test_acc * 100, 2)))
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a9f5b9d-c906-4cf8-921d-02b7638ace87" width="500" height="220" >
</p>

