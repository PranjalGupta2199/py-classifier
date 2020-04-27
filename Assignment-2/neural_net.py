from abc import ABC, abstractmethod
import pickle
from itertools import tee, islice, chain

import pandas as pd
import numpy as np
from numpy.random import uniform, random, normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ActivationFunction(ABC):
    def __init__(self):
        self.prev = -1

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        raise NotImplementedError

class Tanh(ActivationFunction):

    def forward(self, w):
        self.prev = w
        return np.tanh(w)

    def backward(self):
        val = self.forward(self.prev)
        return 1.0 - np.tanh(val) ** 2

class Sigmoid(ActivationFunction):

    def forward(self, w):
        self.prev = w
        return 1/(1 + np.exp(-w))

    def backward(self):
        val = self.forward(self.prev)
        return val * (1 - val)

class Softmax(ActivationFunction):

    def forward(self, w):
        self.prev = w
        numerator_ = np.exp(w)
        denom_ = np.sum(numerator_, axis=1, keepdims=True)
        return (numerator_ / denom_)

    def backward(self):
        pass

class ReLU(ActivationFunction):

    def forward(self, w):
        self.prev = w
        return np.maximum(0, w)

    def backward(self):
        self.prev[self.prev <= 0] = 0
        self.prev[self.prev > 0] = 1
        return self.prev

def get_actv_fn(fn):
    fns = {
        'relu' : ReLU(),
        'softmax' : Softmax(),
        'tanh' : Tanh(),
        'sigmoid': Sigmoid(),
    }
    return fns[fn]

class Initializer:

    init = {
        'uniform': uniform,
        'gaussian' : normal,
        'random' : random,
        'one': np.ones,
        'zero': np.zeros,
    }

    @staticmethod
    def get(fn, shape, a=-1, b=1):
        if fn in ('one', 'zero', 'random'):
            return Initializer.init[fn](shape)
        else:
            return Initializer.init[fn](a, b, shape)

class Layer:

    def __init__(self, input_dim, output_dim, init_method, activation_fn):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._init_weight(init_method, activation_fn)

    def _init_weight(self, init_method, activation_fn):
        self.w = Initializer.get(init_method, (self.output_dim, self.input_dim), 0, 1)
        self.b = np.ones((1, self.output_dim))
        self.activation_fn = get_actv_fn(activation_fn)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        batch_size, _ = X.shape
        self.a = self.activation_fn(np.dot(self.w, X.T).reshape(batch_size, -1) +
                                           self.b)
        return self.a

    def backward(self, a_prev, grad_, lamda):
        fn_grad = self.activation_fn.backward()
        # print ("\t\tFunction Gradient {}".format(fn_grad.shape))
        # print ("\t\tw update {}".format(np.dot(grad_.T, a_prev).shape))
        grad_ = grad_ * fn_grad

        self.w = self.w - lamda * np.dot(grad_.T, a_prev)
        self.b = self.b - lamda * np.sum(grad_, axis=0, keepdims=True)

        return np.dot(grad_, self.w)

    def __str__(self, indent='\t'):
        string = ''
        string += "({})\n".format(__class__.__name__)
        string += indent * 2 + "Weight : {}\n".format(self.w.shape)
        string += indent * 2 + "Output dim : {}\n".format(self.output_dim)
        string += indent * 2 + "Bias : {}\n".format(self.b.shape)
        string += indent * 2 + "Activation : {}\n".format(self.activation_fn.__class__.__name__)
        return string


class NeuralNet:

    def __init__(self, learning_rate, layer_dims, actn_fn, initializer):
        super().__init__()
        self.initializer = initializer
        self.lamda = learning_rate
        self.init_model_params(layer_dims, actn_fn)

    def init_model_params(self, layer_dims, activation_fn):
        next_, prev_ = tee(layer_dims, 2)
        next_ = islice(next_, 1, None)

        self.layers = [
            Layer(
                input_dim=i,
                output_dim=j,
                activation_fn=fn,
                init_method=self.initializer,
            )
            for i,j,fn in zip(prev_,
                              next_,
                              activation_fn)
        ]

    @staticmethod
    def load_state_dict(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        # print ("Forward Propagation :")
        for layer in self.layers:
            # print ("\tinput {}".format(X.shape))
            X = layer(X)
        # print ("\tinput {}".format(X.shape))
        return X

    def backward(self, X, y_true, y_pred):
        # print ("Back Propagation :")
        grad = -2.0 * (y_true - y_pred)
        curr, prev = reversed(self.layers), reversed(self.layers)
        prev = islice(prev, 1, None)

        for curr_layer, prev_layer in zip(curr, prev):
            # print ("\tgrad {}".format(grad.shape))
            # print ("\t\tprev a {}".format(prev_layer.a.shape))
            grad = curr_layer.backward(prev_layer.a, grad, self.lamda)

        # print ("\tgrad {}".format(grad.shape))
        # print ("\t\tprev a {}".format(X.shape))
        self.layers[0].backward(X, grad, self.lamda)

    @property
    def settings(self):
        string = 'Neural Net Architecture:\n'
        for idx, layer in enumerate(self.layers):
            string += "\t{}: {}".format(idx, layer.__str__())
        return string

    def __str__(self):
        return self.settings


def train():
    n = NeuralNet(
            learning_rate=0.1,
            layer_dims=[10, 1],
            actn_fn=["sigmoid"] * 4 ,
            initializer='uniform'
    )
    data = pd.read_csv('housepricedata.csv')
    headers = data.columns
    X = StandardScaler().fit_transform(data[headers[:-1]])[:100, :]
    Y = data[headers[-1]].to_numpy().reshape(-1, 1)[:100, :]

    n_epoch = 50000
    print_batch = 1000
    for n_iter in range(n_epoch):
        y_pred = n(X)
        # print (y_pred.shape)
        n.backward(X, Y, y_pred)
        if n_iter % print_batch == 0:
            print (np.sum(np.square(Y - y_pred)))

    for j, k in zip(Y,n(X)):
        print (j, k)

if __name__ == "__main__":
    train()