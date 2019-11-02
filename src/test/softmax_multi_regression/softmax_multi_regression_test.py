import numpy as np
import unittest
from softmax_regression.softmax_regression_with_SGD import one_in_k_encoding, softmax, SoftmaxClassifier, numerical_grad_check


class SoftTest(unittest.TestCase):
    @staticmethod
    def test_encoding():
        print('*' * 10, 'test encoding')
        labels = np.array([0, 2, 1, 1])
        m = one_in_k_encoding(labels, 3)
        print(m)
        res = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
        assert res.shape == m.shape, 'encoding shape mismatch'
        assert np.allclose(m, res), m - res
        print('Test Passed')

    @staticmethod
    def test_softmax():
        print('Test softmax')
        X = np.zeros((3, 2))
        X[0, 0] = np.log(4)
        X[1, 1] = np.log(2)
        print('Input to Softmax: \n', X)
        sm = softmax(X)
        expected = np.array([[4.0 / 5.0, 1.0 / 5.0], [1.0 / 3.0, 2.0 / 3.0], [0.5, 0.5]])
        print('Result of softmax: \n', sm)
        assert np.allclose(expected, sm), 'Expected {0} - got {1}'.format(expected, sm)
        print('Test complete')

    @staticmethod
    def test_cost():
        print('*' * 5, 'Testing  Cost')
        X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
        w = np.ones((2, 3))
        print("w is;", w)
        y = np.array([0, 1, 2])
        scl = SoftmaxClassifier(num_classes=3)
        x, y = scl.cost_grad(X, y, w)
        expected = -np.log(1 / 3)
        assert np.allclose(x, expected), "Expected {0} - got {1}".format(expected, x)
        print('Test Success')

    @staticmethod
    def test_grad():
        print('*' * 5, 'Testing  Gradient')
        X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
        w = np.ones((2, 3))
        y = np.array([0, 1, 2])
        scl = SoftmaxClassifier(num_classes=3)
        f = lambda z: scl.cost_grad(X, y, W=z)
        numerical_grad_check(f, w)
        print('Test Success')
