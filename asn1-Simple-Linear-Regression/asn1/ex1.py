import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    population_data = [
        [1, 3.5],
        [1, 7],
        ]

    house_data = [
        [1, 1600, 3],
        [1, 2100, 2],
    ]

    print "Running Univariate Linear Regression:\n"
    run_linear_regression('/data/ex1data1.txt', population_data)

    print "Running Multiivariate Linear Regression:\n"
    run_linear_regression('/data/ex1data2.txt', house_data, 'multi')

def run_linear_regression(path, data, lr_type='uni'):
    X, y, theta = preprocess_data(path)

    # best alpha/iters found with find_best_learning_rate
    alpha = 0.01
    iters = 1750

    theta, cost = compute_gradient_descent(X, y, theta, alpha, iters)
    run_predictions(theta, data, lr_type)


def preprocess_data(path_arg):
    data = get_data(path_arg)

    if path_arg == '/data/ex1data2.txt':
        print "Normalizing data for multivariate linear regression"
        data = (data - data.mean()) / data.std()

    # append a 'ones' column to the front of the dataset
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    num_cols = data.shape[1]

    # the splicing is done this way for general-purpose. However many columns
    # our X data has, splice it up until the last column
    # (up to but not including)
    X = data.iloc[:, :num_cols - 1]

    # y will always be the last column, whatever the 'last' is in any general
    # case. Splice from the last column up to the end
    y = data.iloc[:, num_cols - 1:]

    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # set theta to accommodate any size of X's features
    theta = np.matrix(np.zeros(num_cols - 1))

    return X, y, theta


def get_data(path_arg):
    pwd = os.getcwd()
    path = pwd + path_arg

    if path_arg == '/data/ex1data1.txt':
        return pd.read_csv(path, header=None, names=['Population', 'Profit'])
    else:
        return pd.read_csv(path, header=None, names=[
                           'Size', 'Bedrooms', 'Price'])


def run_predictions(theta, data, lr_type='uni'):
    # returns Numpy matrices. Use .item() to retrive integer value
    prediction_1 = compute_hypothesis(data[0], theta).item()
    prediction_2 = compute_hypothesis(data[1], theta).item()

    if lr_type == 'uni':
        print "Population = 35,000, predicted profit is: ${}".format(prediction_1 * 10000)
        print "Population = 70,000, predicted profit is: ${}".format(prediction_2 * 10000)
    else:
        print "Size = 1600ft, bedrooms = 3, predicted price is: ${}".format(prediction_1)
        print "Size = 2100ft, bedrooms = 2, predicted price is: ${}".format(prediction_2)


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
    print "Gradient Descent at alpha = {}, iters = {}".format(alpha, iters)

    m = len(X)
    temp_theta = np.matrix(np.zeros(theta.shape))

    # Unroll your theta parameters to a one-dimensional array. Not necessary
    # for this example, but standard practice for when theta becomes
    # multidimensional (think CNNs, where it's a matrix instead of a vector)
    num_parameters = theta.flatten().shape[1]
    cost_history = np.zeros(iters)

    for i in range(iters):
        h = compute_hypothesis(X, theta)
        temp_theta = theta - (alpha * (1.0 / m)) * sum(np.multiply(h - y, X))

        theta = temp_theta
        cost_history[i] = compute_cost(X, y, theta)

    print "Theta: {}, {}".format(theta[0, 0], theta[0, 1])
    print "Cost: {}\n".format(cost_history[-1])

    return theta, cost_history


def find_best_learning_rate(X, y, theta):
    print "Looking for best learning parameters..."

    alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    iterations = [100, 250, 500, 750, 1000, 1250, 1500, 1750]

    best_alpha = alphas[0]
    best_iters = iterations[0]
    lowest_cost = np.inf

    for alpha in alphas:
        for iters in iterations:
            theta, cost_history = compute_gradient_descent(
                                    X, y, theta, alpha, iters)
            cost = cost_history[-1]

            if cost < lowest_cost:
                lowest_cost = cost
                best_alpha = alpha
                best_iters = iters


    print "Done!\n"
    print "Lowest cost: {}".format(lowest_cost)
    print "Best alpha: {}".format(best_alpha)
    print "Best amount of iterations: {}".format(best_iters)

    return best_alpha, best_iters, lowest_cost


if __name__ == '__main__':
    main()
