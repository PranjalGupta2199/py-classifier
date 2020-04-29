from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils import metrics, min_max_normalize, standardize

class LogisticRegression:
    """Logistic Regression Abstraction"""
    def __init__(self, n_init, n_epoch, learning_rate, beta, reg=None, initializer='zeros'):
        """Init
        :param
            n_init: int
                Number of models to compare
            n_epoch: int
                Number of iterations in each model
            learning_rate: float
                Rate at the which the steps are taken for optimization
                algorithms
            beta: float
                Value which stresses the importance of regularization
            initializer: string
                String value indicating which Initializer to use"""
        self.n_init = n_init
        self.n_epoch = n_epoch
        self.ieta = learning_rate
        self.beta = beta
        self.initializer = initializer
        self.reg = "None" if reg is None else reg.capitalize()

    def _init_weight(self, n_feature):
        """Initializes weights of the model.
        W: Weight matrix (used for dot product with X)
        B: Bias"""
        if self.initializer == 'zeros':
            self.w = np.zeros((1, n_feature))
            self.b = 0.0
        elif self.initializer == 'ones':
            self.w = np.ones((1, n_feature))
            self.b = 1.0
        elif self.initializer == 'random':
            self.w = np.random.random((1, n_feature))
            self.b = np.random.random((1)).item()
        elif self.initializer == 'uniform':
            self.w = np.random.uniform(0, 1, (1, n_feature))
            self.b = np.random.uniform(0, 1, (1)).item()
        elif self.initializer == 'gaussian':
            self.w = np.random.normal(0, 1, (1, n_feature))
            self.b = np.random.normal(0, 1, (1)).item()
        else:
            raise RuntimeError("No Initialiser Found !!")

    def _load(self, path):
        """Loads the data from txt file and splits it into
        training and testing components"""
        data = np.loadtxt(path, delimiter=',')
        return train_test_split(data[:, :-1],
                                np.reshape(data[:, -1], newshape=(-1)),
                                test_size=0.2)

    def _save(self):
        """Saves the model in a binary file"""
        state_dict = {
            'w' : self.w,
            'b' : self.b,
        }
        with open('log_regr_model.npy', 'wb') as f:
            pickle.dump(state_dict, f)

    def load_state_dict(self):
        """Loads binary model file"""
        with open('log_regr_model.npy', 'rb') as f:
            return pickle.load(f)

    def sigmoid(self, val):
        """Sigmoid function implementation"""
        return 1/(1 + np.exp(-val))

    def _train(self, X, Y, print_after, plot):
        """Training function
        :param
            X: numpy array
               Training Examples
            Y: numpy array
                Target values
            print_after: int
                Logs the loss, accuracy after certain iterations
            plot: bool
                Indicates whether to draw the plots or not

        :returns
            cost: float
                Error value after training the model
            accuracy_to_plot: list<float>
                List of accuracies at each iteration of the model
            errors_to_plot: list<float>
                List of errors at each iteration of the model"""
        m, n = X.shape
        temp_error = []
        temp_accuracy = []

        for epoch in range (0, self.n_epoch + 1):
            output = self.sigmoid(np.dot(self.w, X.T) + self.b)

            pos_class_loss = Y * np.log(output)
            neg_class_loss = (1 - Y) * np.log(1 - output)

            cost = (1/m)*np.sum(-pos_class_loss - neg_class_loss)
            dw = (1/m)*np.dot(output - Y, X)
            db = (1/m)*np.sum(output - Y)

            if self.reg == 'L1':
                cost += (self.beta/m)*np.sum(np.abs(self.w))
                dw +=  (self.beta/m)*np.sign(self.w)

            elif self.reg == 'L2':
                cost += (self.beta/m)*np.sum(np.square(self.w))
                dw += (self.beta/m)*self.w

            if plot:
                Y_pred = self.classify(X)
                accuracy = metrics(Y.reshape(-1), Y_pred.reshape(-1))['accuracy']
                temp_error.append(cost)
                temp_accuracy.append(accuracy)

            self.w -= dw*self.ieta
            self.b -= db*self.ieta

            if epoch % print_after == 0:
                print ("\tEpoch {:4}/{} ---> {:.4f} | Accuracy: ---> {:.4f}".format(
                        epoch, self.n_epoch, cost, self.score(Y, output)))

        return cost, temp_accuracy, temp_error

    def fit(self, path, print_after=1, plot=False):
        """Wrapper method for training and saving the model"""
        X_train, X_test, Y_train, Y_test = self._load(path)

        Y_train = Y_train.reshape((1,-1))
        Y_test = Y_test.reshape((1,-1))

        X_train = standardize(X_train)
        X_test = standardize(X_test)

        _, n_feature = X_train.shape

        accuracy_to_plot = []
        error_to_plot = []

        curr_best = -1
        for iter_ in range(self.n_init):
            print ("Running Model {}".format(iter_ + 1))
            self._init_weight(n_feature)
            cost, accuracy, error = self._train(X_train, Y_train, print_after, plot)

            if iter_ == 0 or cost < curr_best:
                self._save()
                curr_best = cost
                accuracy_to_plot = accuracy
                error_to_plot = error

        print ("Loading the best model ...")
        dict_ = self.load_state_dict()
        self.w = dict_['w']
        self.b = dict_['b']

        if plot:
            plt.figure(1)
            plt.plot(range(self.n_epoch + 1), error_to_plot, c='b')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Logistic Loss')
            plt.title('Loss Function vs Epochs')
            plt.savefig('./regr_error_plot.png')

            plt.figure(2)
            plt.plot(range(self.n_epoch + 1), accuracy_to_plot, c='r')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Accuracy %')
            plt.title('Accuracy vs Epochs')
            plt.savefig('./regr_accuracy_plot.png')


        Y_pred = self.classify(X_test)
        dict_ = metrics(Y_test.reshape(-1), Y_pred.reshape(-1))
        print ("Validation Accuracy: {:4}".format(dict_['accuracy']))
        print ("F-Score: {:4}".format(100 * dict_['f1-score']))

    @property
    def settings(self):
        """Returns the string representation of the model"""
        string = 'Model Parameters : \n'
        string += '\tIntialization Techinque : \'{}\'\n'.format(self.initializer)
        string += '\tRegularisation Used : {}\n'.format(self.reg)
        string += '\tWeight (W): {}\n'.format(self.w)
        string += '\tBias (b): {}\n'.format(self.b)
        string += 'Learning rates : \n'
        string += '\t Alpha : {}\n'.format(self.ieta)
        string += '\t Beta : {}\n'.format(self.beta)
        return string

    def _predict(self, Y):
        """Predicts the value passed to the model"""
        labels = np.zeros(Y.shape)
        for idx in range(Y.shape[1]):
            prob = Y[0, idx]
            if prob > 0.5:
                labels[0, idx] = 1
        return labels

    def classify(self, X):
        """Wrapper function to get the predicted value
        for testing example"""
        y_pred = self._predict(self.sigmoid(np.dot(self.w, X.T) + self.b))
        return y_pred


    def score(self, target, output):
        """Returns the Accuracy of the model"""
        y_pred = self._predict(output)
        return metrics(target.reshape(-1), y_pred.reshape(-1))['accuracy']

if __name__ == '__main__':
    reg = LogisticRegression(
                    n_init=10,
                    n_epoch=5000,
                    learning_rate=0.1,
                    beta=0.1,
                    reg='None',
                    initializer='gaussian'
        )
    reg.fit('./data_banknote_authentication.txt', 1000, plot=True)
    print (reg.settings)