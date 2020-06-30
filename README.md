## Description
Neural Network which categorizes an image of an item of clothing into one of 10 categories:
1. T-shirt/top
1. Trouser
1. Pullover
1. Dress
1. Coat
1. Sandal
1. Shirt
1. Sneaker
1. Bag
1. Ankle boot

The network consists of 1 hidden layer of 128 neurons (using the ReLU activation function), and creates an extra softmax layer for predictions.

One can save the trained weights of the model and load them later when needed.

Test accuracy achieved: 0.88

## Installation and running
Uses `Python 3.8` and `Tensorflow` library which can be installed using `pip install tensorflow` along with `numpy` which is installed automatically by Tensorflow (otherwise use `pip install numpy`) and `matplotlib` which comes in-built in Python.

Program can be run using `python main.py`

## Dataset
[MNIST fashion dataset](https://github.com/zalandoresearch/fashion-mnist) included in `Tensorflow`. It contains 60,000 labelled images as training data and 10,000 labelled images as testing data.

Images are 28x28 pixel intensity values (0-255) which are converted into 0-1 floating point values.
