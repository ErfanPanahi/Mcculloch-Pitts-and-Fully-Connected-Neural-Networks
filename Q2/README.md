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

The data for both groups is shown in the figure below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/25db07b6-bfc9-4b6f-871e-e358afbc64e5" width="600" height="500" >
</p>

Now, following the AdaLine learning algorithm, we first initialize the weights, bias, and learning rate with random and small values, and then update the weights and biases until the cost function decreases below a predetermined threshold. The changes in the cost function based on the samples are shown in the figure below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/33f07ee9-ba1d-4b51-8c6b-b4e50258230f" width="600" height="500" >
</p>

As you can see, the cost function has decreased as learning progresses. The method of separating the two classes using the AdaLine network is shown in the figure below. ($accuracy=100$%)

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

We want to test the designed AdaLine network on the new data whose variance has increased. The result is shown in the figure below. As expected, the separation for the new data also has suitable accuracy. ($accuracy = 96$%)

## MAdaLine Neural Network


<p align="center">
  <img src="https://github.com/user-attachments/assets/abc7f3b0-a5a9-4794-a27d-e04bdcee7d04" width="600" height="500" >
</p>

