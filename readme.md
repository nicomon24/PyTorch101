# PyTorch101
This is a simple repository of my first experiments with the PyTorch package, following some of the tutorial from [here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).
I'm already experienced with Tensorflow, so I will focus also on the differences between the two packages.

This series of tutorial is intended only to be of practical usage, not a theoretical analysis of different problems.

## Part 1 - Using Tensors
In this first part we will approach tensor definition and initialization, basic operations like addition and reshaping, plus some cool features like numpy integration and CUDA usage.

[Notebook](PyTorch101%20-%20Part%201%20-%20Using%20Tensors.ipynb)

[PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)

## Part 2 - Autograd
In the second part of the tutorial we explore pytorch's automatic differentiation and we show some of its usage.

[Notebook](PyTorch101%20-%20Part%202%20-%20Autograd.ipynb)

[PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

## Part 3 - Neural Networks
We are going to discuss how to implement the basic components of a neural network, while also providing an introduction to its training.

[Notebook](PyTorch101%20-%20Part%202%20-%20Neural%20Networks.ipynb)

[PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

## Part 4 - MNIST Training
We take a step further in the Neural Networks world, by training our previously defined CNN classifier on the MNIST dataset for simplicity. We do not use CIFAR10 as in the official tutorial because of unavailability of a local GPU (see Part 5 for more details)

[Notebook](PyTorch101%20-%20Part%202%20-%20MNIST%20Training.ipynb)

[PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## Part 5 - Remote GPU Training
This time, we are going to tackle a very similar problem as before, image classification on the MNIST dataset, but diving deeply into GPU training on a remote machine, also demonstrating how we can move trained models around, following the instructions from [here](https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch).

[Notebook](PyTorch101%20-%20Part%202%20-%20Remote%20GPU%20Training.ipynb)

[PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
