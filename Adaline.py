import numpy as np


class GradientDescent(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    --------------------
    eta : float.
        Learning rate (between 0.0 and 0.1)
    n : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    --------------------
    w : 1d-array
        Weights after fitting.
    cost : list
        Sum-of-Squares cost function value in each epoch.
    """
    def __init__(self, eta=0.1, n= 50, random_state=1):
        self.eta = eta
        self.n = n
        self.random_state = random_state
    

    def fit(self, X, y):
        """Fit the training data.
        
        Parameters
        --------------------
        X : {array-like}, shape = [samples, features]
            Training vectors.
        y : {array-like}, shape = [samples]
            Target values.
            
            Returns
            --------------------
            self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost = []
        for i in range(self.n):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self
    

    def net_input(self, X):
        """Calculate the net input."""
        return np.dot(X, self.w[1:]) + self.w[0]
    

    def activation(self, X):
        """Compute linear activation."""
        return X
    

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
