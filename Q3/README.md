In this question, we want to solve a classification problem using an **Auto-Encoder**. For a better understanding of Auto-Encoders, it is recommended to read the attached paper [`liu2017.pdf`](https://github.com/ErfanPanahi/Mcculloch-Pitts-and-Fully-Connected-Neural-Networks/blob/main/Q3/liu2017.pdf). The goal of this exercise is to familiarize yourself with the `keras` package and work with the `MNIST` dataset.

## Introduction to MNIST Dataset

In this section, the goal is to get familiar with and work with the dataset. You can add the dataset using the `torchvision` package as shown below. Alternatively, you can use the attached file [`mnist.npz`](https://github.com/ErfanPanahi/Mcculloch-Pitts-and-Fully-Connected-Neural-Networks/blob/main/Q3/mnist.npz).

```python
import torchvision
torchvision.datasets.MNIST (...)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
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

The figure below shows the architecture of both parts.


<p align="center">
  <img src="https://github.com/user-attachments/assets/42db361a-2f04-420b-beda-6b5b66852578" width="500" height="600" >
</p>

Now, we will train the network. For this purpose, we will use the `adam` optimizer and the `binary_crossentropy` loss function. 

The figure below shows the loss function plot against the number of epochs. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4da049f-592b-4621-aa79-558987412f4b" width="850" height=350" >
</p>

The figure below compares the original images with the decoded ones (the network's output).

<p align="center">
  <img src="https://github.com/user-attachments/assets/10cef668-b49f-4822-9822-59ce7d45bfa5" width="1000" height="450" >
</p>

## Classification

In this section, I want to use a 30-dimensional feature space (the output of the encoder) to build a simple classifier with two hidden layers. To do this, I will separate the encoder part after training the autoencoder and use its outputs to train the classifier.

The figure below shows the architecture of the classifier network


<p align="center">
  <img src="https://github.com/user-attachments/assets/e0d4518a-fa6d-4783-9b6b-374d44955875" width="500" height="150" >
</p>

Now, I will train the classifier network. The figures below show the loss plot and the accuracy plot against the number of epochs for both the training and validation datasets.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a3974e06-2f89-442d-9782-a3afb82644ce" width="850" height="350" >
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/cdc8e674-a027-4a00-9f96-10215d1391bd" width="850" height="350" >
</p>

Finally, we connect the encoder network to the classifier network. 

The figure below shows the architecture of the final network along with the accuracy on the training and test datasets.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a9f5b9d-c906-4cf8-921d-02b7638ace87" width="500" height="220" >
</p>

