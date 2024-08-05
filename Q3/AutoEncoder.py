import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.utils import to_categorical

def train_net(model, train_data, train_labels, test_data, test_labels, Epoch_num):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    History = model.fit(train_data, train_labels, 
                             epochs = Epoch_num,
                             batch_size = 256,
                             validation_data = (test_data, test_labels))
    return History


def mlp_2h_layer(layers_shape):
    clf_input = Input(shape = (layers_shape[0], ))
    clf = clf_input
    clf = Dense(layers_shape[1], activation='relu')(clf)
    clf = Dense(layers_shape[2], activation='relu')(clf)
    clf = Dense(layers_shape[3], activation='sigmoid')(clf)
    classifier = Model(clf_input, clf, name = 'classifier')
    return classifier

def visual_accuracy(acc, val_acc):
    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(val_acc)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Validation Accuracy')

def visual_loss(loss, val_loss):
    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Validation Loss')

def show_decode_images(images, decoded_images, labels):
    plt.figure(figsize = (20,16))
    for i in range(5):
        rnd = np.random.randint(0, len(labels))
        plt.subplot(2, 5, i+1)
        plt.imshow(images[rnd])
        plt.title('number: {} (Input)'.format(labels[rnd]))
        plt.subplot(1, 5, i+1)
        plt.imshow(decoded_images[rnd])
        plt.title('number: {} (Decoded)'.format(labels[rnd]))

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

def load_data():
    path = 'C:/Users/ASUS/Desktop/Deep Learning/Homeworks/HW1/mnist.npz'

    with np.load(path, allow_pickle=True) as f:
        train_images, train_labels = f['x_train'], f['y_train']
        test_images, test_labels = f['x_test'], f['y_test']

    return train_images, train_labels, test_images, test_labels

def data_per_label(data):
    count = np.zeros(10)
    for i in range(10):
        count[i] = sum(data == i)
    get_colors = lambda n: ["#%06x" % np.random.randint(0, 0xFFFFFF) for _ in range(n)]
    plt.bar(range(10), count, color = get_colors(10))
    plt.xticks(range(10))
    plt.ylabel('count')
    plt.xlabel('label')
    plt.title('Number of data per label')

def random_images(train_images, train_labels):
    plt.figure(figsize = (20,16))
    for i in range(5):
        rnd = np.random.randint(0, len(train_labels))
        plt.subplot(1, 5, i+1)
        plt.imshow(train_images[rnd])
        plt.title('number: {}'.format(train_labels[rnd]))
        rnd = np.random.randint(0, len(train_labels))
        plt.subplot(2, 5, i+1)
        plt.imshow(train_images[rnd])
        plt.title('number: {}'.format(train_labels[rnd]))

