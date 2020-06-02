import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

class Perceptron(object):
    """Perceptron classifier
    
    Parameters
    --------------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    --------------------
    w : 1D-array
        Weights after fitting.
    errors : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n=50, random_state=1):
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
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors = []
        for i in range(self.n):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X, self.w[1:] + self.w[0])
    
    def predict(self, X):
        """Return class label  after unit step."""
        return np.where(self.net_input(X) >= 0.0,1,-1)

    def plot_precision_regions(X, y, classifier, resolution=0.02):
        """Method utilized to visualize the decision boundaries for two dimensional datasets."""
        pass
