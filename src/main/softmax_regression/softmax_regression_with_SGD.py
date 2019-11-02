"""多分类SGD
@author: hehaoming
@time: 2019/11/2 11:03
"""

import numpy as np
from commons.read_data import read_data_from_resource


def numerical_grad_check(f, x):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-4
    # d = x.shape[0]
    cost, grad = f(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        dim = it.multi_index
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus - cminus) / (2 * h)
        print('grad, num_grad, grad-num_grad', grad[dim], num_grad, grad[dim] - num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim,num_grad,grad[dim])
        it.iternext()


def softmax(X):
    """
    Compute the softmax of each row of an input matrix (2D numpy array).

    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate

    More precisely this is what you must do.

    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that

    You can do all of it without for loops using numpys vectorized operations.
    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    ### YOUR CODE HERE no for loops please
    e = np.exp(X)
    denoms = sum(e.T)
    res = np.array([e[i] / denoms[i] for i in range(len(X))])
    ### END CODE
    return res


def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes

    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc


class SoftmaxClassifier():

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.W = None

    def cost_grad(self, X, y, W, calculat_grad=1):
        """
        Compute the average negative log likelihood cost and the gradient under the softmax model
        using data X, Y and weight vector W.

        the functions np.log, np.nonzero, np.sum, np.dot (@), may come in handy
        Args:
           X: numpy array shape (n, d) float - the data each row is a data point
           y: numpy array shape (n, ) int - target values in 0,1,...,k-1
           W: numpy array shape (d x K) float - weight matrix
        Returns:
            totalcost: Average Negative Log Likelihood of w
            gradient: The gradient of the average Negative Log Likelihood at w
        """
        cost = np.nan
        grad = np.zeros(W.shape) * np.nan
        Yk = one_in_k_encoding(y, self.num_classes)  # may help - otherwise you may remove it
        ### YOUR CODE HERE
        p = X.dot(W)
        s = softmax(p)
        cost = - 1 / len(y) * sum(sum(Yk * np.log(s)))
        if calculat_grad:
            grad = - 1 / len(y) * X.T.dot(Yk - s)
        ### END CODE
        return cost, grad

    def fit(self, X, Y, W=None, lr=0.01, epochs=10, batch_size=16):
        """
        Run Mini-Batch Gradient Descent on data X,Y to minimize the in sample error (1/n)NLL for softmax regression.
        Printing the performance every epoch is a good idea to see if the algorithm is working

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
           W: numpy array shape (d x K)
           lr: scalar - initial learning rate
           batchsize: scalar - size of mini-batch
           epochs: scalar - number of iterations through the data to use
        Sets:
           W: numpy array shape (d, K) learned weight vector matrix  W
           history: list/np.array len epochs - value of cost function after every epoch. You know for plotting
        """
        b = batch_size
        if W is None: W = np.zeros((X.shape[1], self.num_classes))
        history = []
        ### YOUR CODE HERE
        for i in range(epochs):
            print("starting epoch: ", i)
            p = np.random.permutation(range(len(Y)))
            for j in range(int(len(Y) / batch_size + 0.5)):
                W = W - lr * self.cost_grad(np.array([X[i] for i in p[b * j: b * (j + 1)]]),
                                            np.array([Y[i] for i in p[b * j: b * (j + 1)]]), W)[1]
            print("calculating history for epoch: ", i)
            history = history + [self.cost_grad(X, Y, W, 0)[0]]
        ### END CODE
        self.W = W
        self.history = history

    def score(self, X, Y):
        """ Compute accuracy of classifier on data X with labels Y
        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
        Returns:
           out: float - mean accuracy
        """

        ### YOUR CODE HERE 1-4 lines
        prediction = self.predict(X)
        out = 1 / len(Y) * sum([(prediction[i] == Y[i]) for i in range(len(Y))])
        ### END CODE
        return out

    def predict(self, X):
        """ Compute classifier prediction on each data point in X
        Args:
           X: numpy array shape (n, d) - the data each row is a data point
        Returns
           out: np.array shape (n, ) - prediction on each data point (number in 0,1,..., num_classes
        """
        ### YOUR CODE HERE - 1-4 lines
        out = np.array([np.argmax(x) for x in softmax(X.dot(self.W))])
        ### END CODE
        return out


if __name__ == "__main__":
    softmaxClassifier = SoftmaxClassifier(3)
    data = read_data_from_resource("dataset2")
    softmaxClassifier.fit(data[0], data[1], epochs=1000, batch_size=1)
    print(softmaxClassifier.score(data[0], data[1]))