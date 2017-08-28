import os
import numpy as np
import pandas as pd
import scipy.optimize as opt


def main():
    run_logistic_regression('/data/ex2data1.txt')


def run_logistic_regression(path):
    X, y, theta = preprocess_data(path)

    predictions = run_predictions(X, y, theta)
    correct = ([1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0
                for (a, b) in zip(predictions, y)])

    # just the author's clever way of representing percentages. Mod
    # denominator here will always be larger than numerator, so the remainder
    # will always be the numerator. No need for float conversions or scaling.
    accuracy = sum(correct) % len(correct)

    print 'Unregularized Logistic Regression Accuracy = {0}%'.format(accuracy)


def preprocess_data(path):
    data = get_data(path)

    # add a ones column. This makes the matrix multiplication work out easier
    data.insert(0, 'Ones', 1)

    # set X and y
    num_cols = data.shape[1]
    X = data.iloc[:, :num_cols - 1]
    y = data.iloc[:, num_cols - 1:]

    # convert from dataframe to to numpy arrays and initialize theta
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(num_cols - 1)

    return X, y, theta

def get_data(path_arg):
    pwd = os.getcwd()
    path = pwd + path_arg

    return pd.read_csv(path, header=None, names=[
                       'Exam 1', 'Exam 2', 'Admitted'])


def run_predictions(X, y, theta):
    theta = np.matrix(theta)
    optimized_theta = np.matrix(find_optimal_theta(X, y, theta))

    probability = compute_hypothesis(X * optimized_theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def find_optimal_theta(X, y, theta):
    # Logistic Regression doesn't use the traditional gradient descent
    # approach of stepping through multiple attempts to lower our cost
    # function. Instead, we'll use a built-in, optimized tool from scipy to
    # find this for us. This is what all practical ML models use for the most
    # finding optimized values of theta.

    # this function gives you the an array of the theta parameters, along with
    # some metadata

    # NOTE: The ordering or parameters for your cost and gradient functions are
    # important here. Should be theta, X and y in that order (see arg list for
    # those two functions)
    result = opt.fmin_tnc(
                        func=compute_cost,
                        x0=theta,
                        fprime=compute_gradient_descent,
                        args=(X, y)
                        )

    return result[0]


# this hypothesis uses the Sigmoid Function.
def compute_hypothesis(z):
    return 1.0 / (1 + np.exp(-z))


def compute_cost(theta, X, y):
    # convert from numpy array to matrix for proper shape (rows, cols)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h = compute_hypothesis(X * theta.T)
    m = len(X)

    return (1.0 / m) * np.sum(np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h)))


def compute_gradient_descent(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)
    h = compute_hypothesis(X * theta.T)

    # unroll theta (no need here, but good practice for when theta is a matrix)
    num_parameters = theta.flatten().shape[1]

    # (3, 100) x (100, 1)
    return np.array((1.0 / m) * (X.T * (h - y)))


if __name__ == '__main__':
    main()
