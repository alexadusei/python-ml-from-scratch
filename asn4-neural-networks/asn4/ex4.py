import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder


def main():
    run_neural_network('/data/ex4data1.mat')


def run_neural_network(path):
    X, y, y_onehot = preprocess_data(path)

    # initial setup
    input_size = 400
    hidden_size = 25
    num_labels = 10
    reg_lambda = 1

    # randomly initialize a theta parameter array of the size of the full
    # network's parameters. This allows us to break the symmetry while
    # training the neural network (makes it easier to train)
    theta_params = (np.random.random(size=hidden_size * (input_size + 1)
                             + num_labels * (hidden_size + 1)) - 0.5) * 0.25

    optimized_theta = train_network_parameters(
                        theta_params,
                        input_size,
                        hidden_size,
                        num_labels,
                        X,
                        y_onehot,
                        reg_lambda
                        )

    run_predictions(X, y, optimized_theta, input_size, hidden_size, num_labels)


def preprocess_data(path):
    data = get_data(path)
    X = data['X']
    y = data['y']

    # One-hot encoding turns a class label n (out of k classes) into a vector
    # of length k where index n is "hot" (1) and the rest are zero.
    # Essentially, if we have a 4, then we make the 4th index 1, and all the
    # other indices 0, and so on.
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)

    return X, y, y_onehot


def get_data(path_arg):
    pwd = os.getcwd()

    return loadmat(pwd + path_arg)


def run_predictions(X, y, theta_params, input_size, hidden_size, num_labels):
    X = np.matrix(X)

    theta1, theta2 = roll_theta(
                        theta_params, input_size, hidden_size, num_labels
                        )

    a1, z2, a2, z3, h = compute_forward_propagation(X, theta1, theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)

    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))

    print 'accuracy = {0}%'.format(accuracy * 100)


def roll_theta(theta_params, input_size, hidden_size, num_labels):
    # we have to reshape the output from the optimizer to match the
    # theta parameter matrix shapes that our network is expecting

    # (25, 401)
    theta1 = (np.matrix(
                    np.reshape(theta_params[:hidden_size * (input_size + 1)],
                               (hidden_size, (input_size + 1)))
                    ))
    # (10, 26)
    theta2 = (np.matrix(
                    np.reshape(theta_params[hidden_size * (input_size + 1):],
                               (num_labels, (hidden_size + 1)))
                    ))

    return theta1, theta2


def compute_hypothesis(z):
    return 1 / (1 + np.exp(-z))


def compute_hypothesis_derivative(z):
    return np.multiply(compute_hypothesis(z), (1 - compute_hypothesis(z)))


# we have two theta matrices for each layer. Theta1 is for the input
# layer to calculate the hidden layer, and Theta2 is for the hidden layer
# to calculate the output layer
def compute_forward_propagation(X, theta1, theta2):
    # theta1 = (25, 401)
    # theta2 = (10, 26)

    m = X.shape[0]

    # add our bias unit to each training example (layer 1)
    a1 = np.insert(X, 0, values=np.ones(m), axis=1) # (5000, 401)

    # we combine our input and theta
    z2 = a1 * theta1.T # (5000, 25)

    # add our bias unit to each trainnig example (layer 2)
    # we activate our input and theta (which is z2), activating it is
    # running it under the sigmoid function, converting z2 to a2

    # (5000, 26)
    a2 = np.insert(compute_hypothesis(z2), 0, values=np.ones(m), axis=1)

    # we combine our hidden layer with theta, creating z3
    z3 = a2 * theta2.T # (5000, 10)

    # final hypothesis, by activating z3 via compute_hypothesis()
    h = compute_hypothesis(z3) #(5000, 10)

    return a1, z2, a2, z3, h


def compute_backpropagation(
    theta_params, input_size, hidden_size, num_labels, X, y, reg_lambda
):

    ### first we'll represent the cost function ###

    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    # it is our hidden layer size * input layer size because each hidden
    # layer node is connected to all 400 input nodes, for each hidden
    # node. That's 400 * 25. These Theta parameters are for ALL edges
    # from one layer to another

    theta1, theta2 = roll_theta(
                        theta_params, input_size, hidden_size, num_labels
                        )

    # run the feed-forward pass
    a1, z2, a2, z3, h = compute_forward_propagation(X, theta1, theta2)
    
    # initializations
    delta_accumulator1 = np.zeros(theta1.shape) # (25, 401)
    delta_accumulator2 = np.zeros(theta2.shape) # (10, 26)


    # Because Y and H are matrices now, we can't use the cost function the 
    # same way we did before, as that was vector multiplication, which returns
    # a scalar value. Matrix multiplication returns a new matrix. All we have
    # to do here is get the sums of the diagonals.

    # Why do we do this? Because we're taking all training examples of Y with
    # respect to the output being k, and multiplying it by all training
    # examples of H with respect to the output being k. In other words
    # (let's see how we did with our hypothesis H in comparison to the correct
    # answer Y, for each class. So check the average with class 1, then class
    # 2, then class 3... etc). We do this with Y' (10 x 5000) and 
    # H (5000 x 10), giving us a (k x k) (or 10x10
    # in this case) matrix. This means that for each row, there is only one
    # desirable output, and that is the kth row and kth column value. If we are
    # looking for how we did predicting a 4, then we'd have to look at row 4,
    # col 4 only. Again, this: "look at the '1' class column-vector for H and
    # the '1' class row-vector for Y'. Vector-multiply these so we get the
    # average of how many times Hk got the same answer as Yk. Put the result in
    # (Y'*H) in (1x1). Do the same for class 2, and put it in (Y'*H) (2x2),
    # and 3 for (3x3) etc...". This makes only the diagonol values in the
    # resulting matrix important.

    # trace() returns the sum of only the diagonal values in a matrix
    # (top-left to bottom-right

    # compute the cost
    J = ((1.0 / m) * (np.trace(-y.T * np.log(h))
                        - np.trace((1 - y).T * np.log(1 - h))))

    # add cost regularization term
    J += ((float(reg_lambda) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2))
                                        + np.sum(np.power(theta2[:, 1:], 2))))

    ### end of cost function, on to backpropagation ###

    for t in range(m):
        # Part 1: run through forwardprop
        a1t = a1[t, :] # (1, 401)
        z2t = z2[t, :] # (1, 25)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :]   # (1, 10)
        yt = y[t, :]   # (1, 10)

        # Part 2: Once you get to the final layer h, see how bad you did by
        # comparing your answer for h to our real 'answer-book' y. We call
        # this our delta layer delta
        d3t = ht - yt # (1, 10)

        # Part 3:
        # See how bad we did for our hidden layers. We can't just compare
        # these to our Y value because they're not based on the final output. 
        # We'll have to "backtrack" to figure out our margin of error for our
        # hidden layers. A good way of thinking about this is with this
        # scenario: "We see our margin of error for our output layers by
        # comparing their answers to our answer-book 'y'. We get the
        # differences here. If the differences are extremely small, then our
        # margin of error for our output layer is small. If we have some
        # neurons that have a large margin of error, then we messed up! The
        # only person to blame right now is the previous neuron(s) that gave
        # this bad neuron its bad answer. We must penalize them"
        # "In order to penalize them, we let these previous neurons know how
        # bad THEY did by showing them the margin of error that we got based
        # on their inputs to us ('us' being the output layer). We share with
        # them how bad we did by giving them their weights multiplied by our
        # margin of error, and multiply this by the derivative sigmoid 
        # function". This process goes on until we get to the last hidden
        # layer, which is going to be Layer #2.

        # Set the remaining deltas per layer with their respective calculations
        # first product is (5000 x 25), second product is (5000 x 25)

        # REMEMBER: np.multiply is element-wise multiplication, while
        # * is matrix multiplication
        z2t = np.insert(z2t, 0, values=np.ones(1)) # (1, 26)
        # (1, 26)

        d2t = np.multiply((theta2.T * d3t.T).T, 
                          compute_hypothesis_derivative(z2t)
                          )

        # Part 4:
        # Now we accumulate our margins of error into one big vector for each
        # layer. We call this the delta accumulator. For each layer, it gets
        # our calculated margin of error vector for said layer and multiplies
        # it by our activation layer. We start our accumulator layer in layer
        # 1, and proceed until we get to to second last layer (we're not
        # including the output layer here), so that is layer L (in this case,
        # that's just layer 1 and layer 2).

        # (25 x 5000) * (5000 x 401)
        # This will be size (25 x 401), as in, it is an average of all the
        # training examples for each neuron unit (25) with each pixel (401).
        # We'll balance out this average by multiplying it by (1/m) later on
        delta_accumulator1 = delta_accumulator1 + (d2t[:, 1:]).T * a1t
        delta_accumulator2 = delta_accumulator2 + d3t.T * a2t

    delta_accumulator1 = delta_accumulator1 / m
    delta_accumulator2 = delta_accumulator2 / m

    # add the gradient regularization term (the /m is here again because we
    # didn't average out the reg term as well)
    delta_accumulator1[:, 1:] = (delta_accumulator1[:, 1:] +
                                (theta1[:, 1:] * reg_lambda) / m)
    delta_accumulator2[:, 1:] = (delta_accumulator2[:, 1:] +
                                (theta2[:, 1:] * reg_lambda) / m)

    # unroll the gradient matrices into a single array
    grad = np.concatenate(
                (np.ravel(delta_accumulator1), np.ravel(delta_accumulator2)))

    return J, grad

def train_network_parameters(
    theta_params, input_size, hidden_size, num_labels, X, y_onehot,
    reg_lambda
):

    # minimize the objective function
    fmin = minimize(
            fun=compute_backpropagation,
            x0=theta_params,
            args=(
                input_size,
                hidden_size,
                num_labels,
                X,
                y_onehot,
                reg_lambda),
            method='TNC',
            jac=True,
            options={'maxiter': 250}
            )

    # return the optimized theta parameters within fmin object
    return fmin.x

if __name__ == '__main__':
    main()
