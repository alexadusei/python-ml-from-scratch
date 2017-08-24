import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    X, y, theta = preprocess_data()

    compute_cost(X, y, theta)


def get_data():
    pwd = os.getcwd()
    path = pwd + '/data/ex1data1.txt'

    return pd.read_csv(path, header=None, names=['Population', 'Profit'])


def preprocess_data():
    data = get_data()

    # append a 'ones' column to the front of the dataset
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]

    # the splicing is done this way for general-purpose. However many columns
    # our X data has, splice it up until the last column
    # (up to but not including)
    X = data.iloc[:, :cols-1]

    # y will always be the last column, whatever the 'last' is in any general
    # case. Splice from the last column up to the end
    y = data.iloc[:, cols-1:]

    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))

    return X, y, theta

def compute_hypothesis(X, theta):
    return X * theta.T

def compute_cost(X, y, theta):
    m = len(X)
    h = compute_hypothesis(X, theta)

    # Remember: Python truncats division. Include decimal (like 1.0) to
    # maintain fractional operations
    cost = (1.0 / (2 * m)) * np.sum(np.power(h - y, 2))

    return cost


def compute_gradient_descent(X, y, theta, alpha=0.01, iters=1500):
    m = len(X)
    h = compute_hypothesis(X, theta)

    pass

if __name__ == '__main__':
    main()