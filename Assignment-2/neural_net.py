from abc import ABC, abstractmethod
import pickle
from itertools import tee, islice, chain

import pandas as pd
import numpy as np
from numpy.random import uniform, random, normal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import standardize, metrics


class ActivationFunction(ABC):
    """Generic Activation Function abstraction."""
    def __init__(self):
        """init
        self.prev stores the last value passed to calculate
        the activation function value.
        This value will be later used for back-propagation."""
        self.prev = -1

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward Propagation"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        """Back Propagation"""
        raise NotImplementedError

class Tanh(ActivationFunction):
    """Tanh Activation Function"""
    def forward(self, w):
        self.prev = w
        return np.tanh(w)

    def backward(self):
        val = self.forward(self.prev)
        return 1.0 - np.tanh(val) ** 2

class Sigmoid(ActivationFunction):
    """Sigmoid Activation Function"""
    def forward(self, w):
        self.prev = w
        return 1/(1 + np.exp(-w))

    def backward(self):
        val = self.forward(self.prev)
        return val * (1 - val)

class ReLU(ActivationFunction):
    """Rectified Linear Unit (ReLU)
    Activation Function"""
    def forward(self, w):
        self.prev = w
        return np.maximum(0, w)

    def backward(self):
        self.prev[self.prev <= 0] = 0
        self.prev[self.prev > 0] = 1
        return self.prev

def get_actv_fn(fn):
    """Returns an instance of an activation function"""
    fns = {
        'relu' : ReLU(),
        'tanh' : Tanh(),
        'sigmoid': Sigmoid(),
    }
    return fns[fn]

class Initializer:
    """Initialization Techniques"""
    init = {
        'uniform': uniform,
        'gaussian' : normal,
        'random' : random,
        'one': np.ones,
        'zero': np.zeros,
    }

    @staticmethod
    def get(fn, shape, a=-1, b=1):
        """Returns initialized values according to
        technique used"""
        if fn in ('one', 'zero', 'random'):
            return Initializer.init[fn](shape)
        else:
            return Initializer.init[fn](a, b, shape)

class Layer:
    """Generic Neural Layer Abstraction"""
    def __init__(self, input_dim, output_dim, init_method, activation_fn):
        """This class transforms the training/testing data
        from input_dim to output_dim followed by the activation
        function used."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._init_weight(init_method, activation_fn)

    def _init_weight(self, init_method, activation_fn):
        """Initializes weights of the layer.
        W: Weight matrix (used for dot product with X)
        B: Bias"""
        self.w = Initializer.get(init_method, (self.output_dim, self.input_dim))
        self.b = Initializer.get(init_method, (self.output_dim, 1))
        self.activation_fn = get_actv_fn(activation_fn)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        """Forward Propagation
        a = h( dot_product (w, X) + bias )"""
        self.a = self.activation_fn(np.dot(self.w, X) + self.b)
        return self.a

    def backward(self, a_prev, upstream_grad, lamda, outer=False):
        """Back Propagation
        Computes local gradient and updates the weights.
        :param
            outer: bool
                Indicates whether the layer is the output layer or not
            upstream_grad: numpy.array
                Gradient passed on from upper layers.
            a_prev:
                Previous layer's pass on value. (Used for updating w)

        :returns
            The new upstream grad which need to be passed
        backward to initial layers of the neural network."""
        if not outer:
            fn_grad = self.activation_fn.backward()
            upstream_grad = upstream_grad * fn_grad

        new_upstream_grad = np.dot(self.w.T, upstream_grad)

        self.w = self.w - lamda * np.dot(upstream_grad, a_prev.T)
        self.b = self.b - lamda * np.sum(upstream_grad, axis=1, keepdims=True)
        return new_upstream_grad

    def __str__(self, indent='\t'):
        """Returns String Representation of the Layer"""
        string = ''
        string += "({})\n".format(__class__.__name__)
        string += indent * 2 + "Transforms : {} ---> {}\n".format(self.input_dim, self.output_dim)
        string += indent * 2 + "Weight : {}\n".format(self.w.shape)
        string += indent * 2 + "Bias : {}\n".format(self.b.shape)
        string += indent * 2 + "Activation : {}\n".format(self.activation_fn.__class__.__name__)
        return string


class NeuralNet:
    """Vanilla Neural Network Implementation"""
    def __init__(self, learning_rate, layer_dims, actn_fn, initializer):
        """Init
        :param
            learning_rate: float
                Used by the optimization algorithm for
                updating weights
            layer_dims: list<int>
                Number of Neurons in each layer including the
                input and output layer
            actn_fn: list<string>
                List of Activation Functions used in each layer
            initializer: string
                Indicates which Intialization technique to use"""
        super().__init__()
        self.initializer = initializer
        self.lamda = learning_rate
        self.init_model_params(layer_dims, actn_fn)

    def init_model_params(self, layer_dims, activation_fn):
        """Compiles the model with all the layers stored in a
        list form."""
        assert len(layer_dims) - 1 == len(activation_fn)
        ## Generate two iterators of the dim list
        next_, prev_ = tee(layer_dims, 2)
        ## skips the first element of the iterator
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
        """Returns the binary saved model"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        """Forward Propagation implementation"""
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, X, y_true, y_pred):
        """Back Propagation implementation"""
        _,n = X.shape
        grad = 1/n * (y_pred - y_true)

        curr, prev = reversed(self.layers), reversed(self.layers)
        prev = islice(prev, 1, None)
        for idx, (curr_layer, prev_layer) in enumerate(zip(curr, prev)):
            if idx == 0:
                grad = curr_layer.backward(prev_layer.a, grad, self.lamda, outer=True)
            else:
                grad = curr_layer.backward(prev_layer.a, grad, self.lamda)

        self.layers[0].backward(X, grad, self.lamda)

    def classify(self, y_pred):
        """Returns the predicted classes from the probability values
        obtained from the final layer"""
        return np.array([1 if i > 0.5 else 0 for i in y_pred])

    def score(self, y_true, y_pred):
        """Returns the accuracy score of the model"""
        return metrics(y_true, y_pred)['accuracy']
        # return accuracy_score(y_true, y_pred)*100

    def error(self, y_true, z):
        """Returns the loss function value of the model"""
        _, n = y_true.shape
        assert y_true.shape == z.shape
        # return np.sum(np.square(y_true - y_pred)) / (2*n)
        return np.sum(np.maximum(z, 0) - z * y_true + np.log(1 + np.exp(-np.abs(z))))/n

    @property
    def settings(self):
        """Returns the string representation of the model
        with each layer showing its configuration"""
        string = 'Neural Net Architecture:\n'
        for idx, layer in enumerate(self.layers):
            string += "\t{}: {}".format(idx, layer.__str__())
        return string

    def save(self, path):
        """Saves the model at the given path"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __str__(self):
        """Returns String representation of the model"""
        return self.settings


def train(plot=False):
    """Training Function"""
    data = pd.read_csv('housepricedata.csv')
    headers = data.columns
    X_train, X_test, Y_train, Y_test = train_test_split(data[headers[:-1]],
                                                        data[headers[-1]])

    X = standardize(X_train[headers[:-1]].to_numpy()).T
    Y = Y_train.to_numpy().reshape(1, -1)

    n_init = 10
    n_epoch = 1000
    print_after = 1000
    curr_best = -1
    save_path = 'nn_model.pickle'

    errors_to_plot = []
    accuracy_to_plot = []

    for _iter in range(n_init):
        print ("Running Model {}: ".format(_iter + 1))
        temp_errors = []
        temp_accuracy = []
        nn = NeuralNet(
                learning_rate=0.1,
                layer_dims=[10, 5, 5, 1],
                actn_fn=['tanh', 'relu', 'sigmoid'],
                initializer='gaussian'
        )
        for _epoch in range(n_epoch):
            output = nn(X)
            y_pred = nn.classify(output.reshape(-1))
            score = metrics(Y.reshape(-1), y_pred)

            temp_errors.append(nn.error(Y, nn.layers[-1].activation_fn.prev))
            temp_accuracy.append(score['accuracy'])
            nn.backward(X, Y, output)

            if (_epoch + 1) % print_after == 0:
                y_pred = nn.classify(output.reshape(-1))
                error_ = nn.error(Y, nn.layers[-1].activation_fn.prev)
                score = metrics(Y.reshape(-1), y_pred)
                print ("\tEpoch {:10}/{} ---> {:.4f} | Accuracy: ---> {:.4f}".format(
                        _epoch + 1, n_epoch, error_, score['accuracy']))


        y_pred = nn.classify(output.reshape(-1))
        curr_result = nn.error(Y, nn.layers[-1].activation_fn.prev)
        if _iter == 0 or curr_result < curr_best:
            curr_best = curr_result
            nn.save(save_path)
            errors_to_plot = temp_errors
            accuracy_to_plot = temp_accuracy

    if plot:
        plt.figure(1)
        plt.plot(range(1, n_epoch + 1), errors_to_plot, c='b')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title('Loss Function vs Epochs')
        plt.savefig('./nerual_net_error_plot.png')

        plt.figure(2)
        plt.plot(range(1, n_epoch + 1), accuracy_to_plot, c='r')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy %')
        plt.title('Accuracy vs Epochs')
        plt.savefig('./neural_net_accuracy_plot.png')

    print ("Loading best model ...")
    nn = NeuralNet.load_state_dict(save_path)
    print (nn)

    X = standardize(X_test[headers[:-1]].to_numpy()).T
    Y = Y_test.to_numpy().reshape(1, -1)
    y_pred = nn.classify(nn(X).reshape(-1))
    scores = metrics(Y.reshape(-1), y_pred)
    print ("Accuracy : {}".format(score['accuracy']))
    print ("F1-Score : {}".format(score['f1-score']))


if __name__ == "__main__":
    train(True)