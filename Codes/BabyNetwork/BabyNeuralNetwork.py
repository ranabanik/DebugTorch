import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from BabyNetwork.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset

np.random.seed(1)

X, Y = load_planar_dataset()
# print(X.shape, Y.shape)
# Y = np.squeeze(Y)
# print(Y)
if __name__ != '__main__':
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
    plt.show()

if __name__ !='__main__':
    clf = LogisticRegressionCV()
    clf.fit(X.T, Y.T)

    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title('Logistic regression')
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
           '% ' + "(percentage of correctly labelled datapoints)")
    plt.show()


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = None
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

if __name__ != '__main__':
    parameters = initialize_parameters(2, 6, 2)
    print(parameters)


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    #     print("Z1",Z1.shape)
    A1 = np.tanh(Z1)
    #     print("A1",A1.shape)
    Z2 = np.dot(W2, A1) + b2
    #     print("Z2",Z2.shape)
    A2 = sigmoid(Z2)
    # print("A2", A2.shape)
    assert (A2.shape == (1, X.shape[1]))  # (1,400)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    assert (Y.shape == (1, Y.shape[1]))
    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))  # =400
    cost = - np.sum(logprobs) / m


    cost = np.squeeze(cost)
    assert (isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']



    dZ2 = A2 - Y  # (1,400)
    dW2 = (np.dot(dZ2, (A1.T))) / m  # (1,4)
    #     dW2 = dW2/m
    db2 = (np.sum(dZ2, axis=1, keepdims=True)) / m
    dZ1 = np.dot((W2.T), dZ2) * (1 - np.power(A1, 2))
    dW1 = (np.dot(dZ1, (X.T))) / m
    db1 = (np.sum(dZ1, axis=1, keepdims=True)) / m
    ### END CODE HERE ###

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]

    n_y = layer_sizes(X, Y)[2]
    # print("start, end", n_x, n_y)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.random.randn(n_h, 1)
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.random.randn(n_y, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}


    for i in range(0, num_iterations):  # epoch -> 0 to 9999

        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    A2, cache = forward_propagation(X, parameters)
    # print("A2: ", A2.shape)
    predictions = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        if A2[0, i] > 0.5:
            predictions[i] = 1

    return predictions

if __name__ != '__main__':
    A2, cache = forward_propagation(X[:, 0:3], initialize_parameters(2, 4, 1))
    print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

if __name__ != '__main__':
    X_test = X[:, 0:3]
    # A2_test, cache = forward_propagation(X_test, initialize_parameters(2, 4, 1))
    # print(A2_test.shape)
    Y_test = Y[:, 0:3]
    # print(Y_test.shape)
    # cost = compute_cost(A2=A2_test, Y=Y_test, parameters=initialize_parameters(2, 4, 1))
    # print(cost)
    # parameters = initialize_parameters(2, 4, 1)
    # grads = backward_propagation(parameters, cache, X_test, Y_test)
    # print("dW1 = " + str(grads["dW1"]))
    # print("db1 = " + str(grads["db1"]))
    # print("dW2 = " + str(grads["dW2"]))
    # print("db2 = " + str(grads["db2"]))

    # parameters = update_parameters(parameters, grads)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))
    print(X_test.shape, Y_test.shape)
    parameters = nn_model(X_test, Y_test, 4, num_iterations=10000, print_cost=True)

    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    preds = predict(parameters, X_test)
    print("predictions mean = " + str(np.mean(preds)))

if __name__ == '__main__':
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(4, 2, i+1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    plt.show()