import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize

def main():
    run_multivariate_logistic_regression('/data/ex3data1.mat')


def run_multivariate_logistic_regression(path):
    X, y = preprocess_data(path)
    all_theta = compute_one_vs_all(X, y, 10, 1)

    run_predictions(X, y, all_theta)


def preprocess_data(path):
    data = get_data(path)

    return data['X'], data['y']


def get_data(path_arg):
    pwd = os.getcwd()
    return loadmat(pwd + path_arg)


def run_predictions(X, y, all_theta):
    y_pred = predict_all(X, all_theta)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct))/ float(len(correct)))

    print 'Accuracy = {0}%'.format(accuracy * 100)


def predict_all(X, all_theta):
    num_examples = X.shape[0]
    num_features = X.shape[1]
    num_labels = all_theta.shape[0]
    
    # same as before, insert ones
    X = np.insert(X, 0, values=np.ones(num_examples), axis=1)
    
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    
    # compute the class probability for each class on each training example
    h = compute_hypothesis(X * all_theta.T)
    
    # create array of the index with maximum probability
    h_argmax = np.argmax(h, axis=1)
    
    # because our array was zero-indexed, we need to add one for the true
    # label prediction
    h_argmax = h_argmax + 1
    
    return h_argmax


def compute_hypothesis(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(theta, X, y, reg_lambda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h = compute_hypothesis(X * theta.T)
    m = len(X)

    # don't regularize the first theta term
    reg = (reg_lambda / (2 * m)) * np.sum(np.power(theta[:, 1:], 2))

    return (((1.0 / m) * np.sum(np.multiply(-y, np.log(h))
                - np.multiply((1 - y), np.log(1 - h)))) + reg)


def compute_gradient_descent(theta, X, y, reg_lambda):
    # import pdb; pdb.set_trace()
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)

    h = compute_hypothesis(X * theta.T)
    
    reg = ((reg_lambda / m) * theta)

    # regularize the whole set
    # WARNING 1: in numpy, np.multiply is thought of as matrix multiplication, 
    # NOT as dot product.

    # WARNING 2: in pandas, adding two matrices with different dimensions
    # undergoes matrix broadcasting, which will reshape the resulting matrix
    # to prevent involuntarily reshaping when adding, always ensure both your
    # matrices are the same dimensiosn when adding
    grad = ((1.0 / m) * (X.T * (h - y))).T + reg

    # update the first term to not be regularized
    grad[0, 0] = (1.0 / m) * np.sum(np.multiply(h - y, X[:, 0]))

    return grad


def compute_one_vs_all(X, y, num_labels, reg_lambda):
    print "Training..."
    
    num_examples = X.shape[0] # 5000
    num_features = X.shape[1] # 400
    
    # k x (n + 1) array for the parameters of each of the k classifiers
    # (10, 401) <- 11 due to 'ones' column we'll be adding next
    # this represents our theta parameters for each label. We'll train our
    # model to learn the difference between a zero and a one.. all the way to
    # nine. The rows are the label while the columns are each pixel (401)
    all_theta = np.zeros((num_labels, num_features + 1))
    
    # insert a column of ones at the beginning for the intercept term
    # (5000, 401)
    X = np.insert(X, 0, values=np.ones(num_examples), axis=1)

    # train theta parameters for each class label
    # y-labels are 1-indexed instead of 0-indexed in Andrew Ng's example, so
    # we'll follow that convention.
    for i in range(1, num_labels + 1):
        theta = np.zeros(num_features + 1) # (401,)
        y_i = np.array([1 if label == i else 0 for label in y]) # (5000,)
        # turns into 2-dimensional ndarray, (5000, 1). 'reshape'() takes 2
        # parameters; the object to reshape, and the dimensions (a tuple)
        y_i = np.reshape(y_i, (num_examples, 1))

        # minimize the cost function for each classifier
        fmin = minimize(
                fun=compute_cost,
                x0=theta,
                args=(X, y_i, reg_lambda),
                method='TNC',
                jac=compute_gradient_descent
                )

        all_theta[i-1, :] = fmin.x

    return all_theta


if __name__ == '__main__':
    main()
