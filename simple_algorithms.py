import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    pass


class Perceptron(object):
    """Perceptron classifier
    This only works for linear distincion, that is if the ai is able to seperate the two in a line
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


def plot_precision_regions(self, X, y, classifier, resolution=0.02):
    """Method utilized to visualize the decision boundaries for two dimensional datasets."""
    #Setup marker generator and color map
    MARKERS = ('s', 'x', 'o', '^', 'v')
    COLORS = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(COLORS[:len(np.unique(y))])
    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx2.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=COLORS[idx], marker=MARKERS[idx], label=cl, edgecolor='black')


if __name__ == "__main__":
    main()