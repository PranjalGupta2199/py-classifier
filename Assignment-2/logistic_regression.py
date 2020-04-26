from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pickle


class LogisticRegression:

    def __init__(self, n_init, n_epoch, learning_rate, beta, reg=None, initializer='zeros'):
        self.n_init = n_init
        self.n_epoch = n_epoch
        self.ieta = learning_rate
        self.beta = beta
        self.initializer = initializer
        self.reg = "None" if reg is None else reg.capitalize()

    def _init_weight(self, n_feature):
        if self.initializer == 'zeros':
            self.w = np.zeros((1, n_feature))
            self.b = 0.0
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
        data = np.loadtxt(path, delimiter=',')
        return train_test_split(data[:, :-1],
                                np.reshape(data[:, -1], newshape=(-1)),
                                test_size=0.2)

    def _save(self):
        state_dict = {
            'w' : self.w,
            'b' : self.b,
        }
        with open('log_regr_model.npy', 'wb') as f:
            pickle.dump(state_dict, f)

    def load_state_dict(self):
        with open('log_regr_model.npy', 'rb') as f:
            return pickle.load(f)

    def sigmoid(self, val):
        return 1/(1 + np.exp(-val))

    def _train(self, X, Y, print_after):
        m, n = X.shape

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

            self.w -= dw*self.ieta
            self.b -= db*self.ieta

            if epoch % print_after == 0:
                print ("\tEpoch {:4}/{} ---> {:.4f} | Accuracy: ---> {:.4f}".format(
                        epoch, self.n_epoch, cost, self.score(Y, output)))

        return cost

    def fit(self, path, print_after=1):
        X_train, X_test, Y_train, Y_test = self._load(path)
        Y_train = Y_train.reshape((1,-1))
        Y_test = Y_test.reshape((1,-1))
        _, n_feature = X_train.shape

        curr_best = -1
        for iter_ in range(self.n_init):
            print ("Running Model {}".format(iter_ + 1))
            self._init_weight(n_feature)
            cost = self._train(X_train, Y_train, print_after)

            if iter_ == 0 or cost < curr_best:
                self._save()
                curr_best = cost

        print ("Loading the best model ...")
        dict_ = self.load_state_dict()
        self.w = dict_['w']
        self.b = dict_['b']

        Y_pred = self.classify(X_test)
        print ("Validation Accuracy: {:4}".format(self.score(Y_test, Y_pred)))
        print ("F-Score: {:4}".format(100 * f1_score(Y_test.reshape(-1), Y_pred.reshape(-1))))

    @property
    def settings(self):
        string = 'Model Parameters : \n'
        string += '\tIntialisation Techinque : \'{}\'\n'.format(self.initializer)
        string += '\tRegularisation Used : {}\n'.format(self.reg)
        string += '\tWeight (W): {}\n'.format(self.w)
        string += '\tBias (b): {}\n'.format(self.b)
        string += 'Learning rates : \n'
        string += '\t Alpha : {}\n'.format(self.ieta)
        string += '\t Beta : {}\n'.format(self.beta)
        return string

    def _predict(self, Y):
        labels = np.zeros(Y.shape)
        for idx in range(Y.shape[1]):
            prob = Y[0, idx]
            if prob > 0.5:
                labels[0, idx] = 1
        return labels

    def classify(self, X):
        y_pred = self._predict(self.sigmoid(np.dot(self.w, X.T) + self.b))
        return y_pred


    def score(self, target, output):
        y_pred = self._predict(output)
        return accuracy_score(target.reshape(-1), y_pred.reshape(-1))*100


if __name__ == '__main__':
    reg = LogisticRegression(10, 2000, 0.01, 0.01, 'l2', 'gaussian')
    reg.fit('./data_banknote_authentication.txt', 100)
    print (reg.settings)