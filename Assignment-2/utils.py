import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler


def min_max_normalize(X):
    """Min-Max normalization function
    X = (X - Xmin)/(Xmax - Xmin)"""
    samples, features = X.shape

    for i in range(features):
        xmin = X[:, i].min()
        xmax = X[:, i].max()
        X[:, i] = (X[:, i] - xmin)/(xmax - xmin)

    return X


def standardize(X):
    """Standardizes the model according to
    the formula
        x = (x - u)/ s
        where u = mean and s = std of feature x"""
    samples, features = X.shape

    for i in range(features):
        u = np.mean(X[:, i])
        std = np.std(X[:, i])
        X[:, i] = (X[:, i] - u)/ std

    return X


def metrics(y_true, y_pred):
    """Calculates the accuracy, f1-score,
    precision and recall of the model"""
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0

    for i, j in zip(y_true, y_pred):
        if (i == 1 and j == 1): tp += 1
        elif (i == 0 and j == 0): tn += 1
        elif (i == 1 and j == 0): fn += 1
        else: fp += 1

    try:
        precision = tp/(tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp/(tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        fscore = (2*precision*recall)/(precision + recall)
    except ZeroDivisionError:
        fscore = 0
    try:
        accuracy = 100 * (tp + tn)/(tp + tn + fp + fn)
    except ZeroDivisionError:
        accuracy = 0

    return ({
        'f1-score': fscore,
        'precision': precision,
        'recall' : recall,
        'accuracy': accuracy,
    })



if __name__ == "__main__":
    a = np.arange(1, 21, dtype=np.float).reshape(-1, 4)
    print (a)
    # print (min_max_normalize(a))
    # print (standardize(a))
    # print (StandardScaler().fit_transform(a))
    actual = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
    print (Counter(zip(actual, predicted)))

    print (metrics(actual, predicted))