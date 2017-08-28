import os
import numpy as np
import pandas as pd
import scipy.optimize as opt


def main():
    run_logistic_regression('/data/ex2data1.txt')
    run_logistic_regression('/data/ex2data2.txt', 'reg')


def run_logistic_regression(path, type='normal'):
    if type == 'normal':
        X, y, theta = preprocess_data(path)
        data_params = (X, y)
    else:
        X, y, theta, reg_lambda = preprocess_data_reg(path)
        data_params = (X, y, reg_lambda)

    predictions = run_predictions(data_params, theta)
    correct = ([1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0
                for (a, b) in zip(predictions, y)])

    # just the author's clever way of representing percentages. Mod
    # denominator here will always be larger than numerator, so the remainder
    # will always be the numerator. No need for float conversions or scaling.
    accuracy = sum(correct) % len(correct)

    print "\n-----------------------------------------------------"
    print 'Unregularized Logistic Regression Accuracy = {0}%'.format(accuracy)
    print "-----------------------------------------------------\n"


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


def preprocess_data_reg(path):
    data = get_data(path, 'reg')

    # because this is a more complicated dataset, there is no linear boundary
    # here to draw a straight line. One way to deal with this is using a
    # complex polynomial with logistic regression to construct features in
    # the shape of a circle. We'll arbitrarily create many higher-dimensional
    # polynomials

    degree = 5
    x1 = data['Test 1']
    x2 = data['Test 2']

    data.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(0, i):
            data['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

    # remove the first two test columns. This makes 'Accepted' the first column
    # and 'Ones' the second column
    data.drop('Test 1', axis=1, inplace=True)
    data.drop('Test 2', axis=1, inplace=True)

    num_cols = data.shape[1]
    X = data.iloc[:, 1:num_cols]
    y = data.iloc[:, :1]

    # convert numpy arrays and initialize parameter array theta
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(num_cols - 1)

    # initialize value for regularization parameber lambda
    reg_lambda = 1

    return X, y, theta, reg_lambda

def get_data(path_arg, type='normal'):
    pwd = os.getcwd()
    path = pwd + path_arg

    if type == 'normal':
        return pd.read_csv(path, header=None, names=[
                           'Exam 1', 'Exam 2', 'Admitted'])
    else:
        return pd.read_csv(path, header=None, names=[
                           'Test 1', 'Test 2', 'Accepted'])


# args here includes X, y, and reg_lambda depending on type of log. regression
def run_predictions(args, theta):
    X = args[0]

    theta = np.matrix(theta)
    optimized_theta = np.matrix(find_optimal_theta(args, theta))

    probability = compute_hypothesis(X * optimized_theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def find_optimal_theta(args, theta):
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
    cost_func = compute_cost if len(args) == 2 else compute_cost_reg
    gradient_func = (compute_gradient_descent if len(args) == 2 else
                    compute_gradient_descent_reg)

    result = opt.fmin_tnc(
                func=cost_func,
                x0=theta,
                fprime=gradient_func,
                args=(args)
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

def compute_cost_reg(theta, X, y, reg_lambda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h = compute_hypothesis(X * theta.T)
    m = len(X)

    # don't regularize the first theta term
    reg = (reg_lambda / (2 * m)) * np.sum(np.power(theta[:, 1:], 2))

    return (((1.0 / m) * np.sum(np.multiply(-y, np.log(h))
                - np.multiply((1 - y), np.log(1 - h)))) + reg)


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


def compute_gradient_descent_reg(theta, X, y, reg_lambda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)

    num_parameters = theta.flatten().shape[1]
    grad = np.zeros(num_parameters)

    for i in range(num_parameters):
        h = compute_hypothesis(X * theta.T)
        term = np.multiply(h - y, X[:, i])

        if i == 0:
            grad[i] = np.sum(term) / m
        else:
            grad[i] = (np.sum(term) / m) + ((reg_lambda / m) * theta[:, i])

    return grad

if __name__ == '__main__':
    main()
