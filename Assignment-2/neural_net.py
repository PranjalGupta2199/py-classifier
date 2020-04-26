from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import uniform, random, normal
from itertools import tee, islice, chain
import pickle
from abc import ABC, abstractmethod


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

    def backward(self, dA, z):
        pass

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
        return np.maximum(0, w)

    def backward(self, dA, w):
        pass

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
        if fn in ('one', 'zero'):
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
        self.w = Initializer.get(init_method, (self.output_dim, self.input_dim))
        self.b = Initializer.get(init_method, (self.output_dim, 1))
        self.activation_fn = get_actv_fn(activation_fn)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        batch_size, _ = X.shape
        self.a = self.activation_fn(np.dot(self.w, X) + self.b)
        # print ("\toutput {}".format(self.a.shape))
        return self.a

    def backward(self, a_prev, grad_, lamda):
        fn_grad = self.activation_fn.backward()
        # print ("\t\tFunction Gradient {}".format(fn_grad.shape))
        # print ("\t\tw update {}".format(np.dot(grad_, a_prev.T).shape))
        grad_ = grad_ * fn_grad

        self.w -= lamda * np.dot(grad_, a_prev.T)
        self.b -= lamda * np.sum(grad_, axis=1, keepdims=True)

        return np.dot(self.w.T, grad_)

    def __str__(self, indent='\t'):
        string = ''
        string += "({})\n".format(__class__.__name__)
        string += indent * 2 + "Weight : {}\n".format(self.w.shape)
        string += indent * 2 + "Bias : {}\n".format(self.b.shape)
        string += indent * 2 + "Activation : {}\n".format(self.activation_fn.__class__.__name__)
        return string


class NeuralNet:

    def __init__(self, learning_rate, layer_dims, actn_fn,
                 n_epoch, n_init, initializer):
        super().__init__()
        self.n_init = n_init
        self.n_epoch = n_epoch
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
                init_method='one',
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
        return self._forward(*args, **kwargs)

    def _forward(self, X):
        # print ("Forward Propagation :")
        for layer in self.layers:
            # print ("\tinput {}".format(X.shape))
            X = layer(X)
        # print ("\tinput {}".format(X.shape))
        return X

    def backward(self, X, y_true, y_pred):
        # print ("Back Propagation :")
        grad = 2 * (y_true - y_pred)
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


if __name__ == "__main__":
    n = NeuralNet(
            learning_rate=0.1,
            layer_dims=[4, 300, 2, 123, 234, 1],
            actn_fn=["sigmoid"] * 6,
            n_epoch=10,
            n_init=1,
            initializer='one'
    )
    x = 2.0 * np.ones((4,2))
    y_pred = n(x)
    n.backward(x, np.ones((1,2)), y_pred)
    print (n)