import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

def main():
    """
    This contains the sample runs from the book Python Machine Learning
    """
    ###
    #Show the last portion of the Dataframe
    ###
    df = pd.read_csv('D:\Github\MLpython\iris.data', header=None, encoding='utf-8')
    df.tail()
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    ##extract sepal length and petal length
    X = df.iloc[0:100, [0,2]].values
    ##plot data
    plt.figure()
    plt.title('Iris Dataset [100]')
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100 , 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.xlabel('petal length [cm]')
    plt.legend(loc='upper left')
    ###
    #Comparing the epochs with the number of updates
    ###
    ppn = Perceptron(eta=0.1, n=10)
    ppn.fit(X,y)
    plt.figure()
    plt.title('Number of Updates vs. Epochs')
    plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
    ###
    #Plot decision regions (show the effect of a perceptron on deciding the difference b/w setosa and versicolor)
    ###
    plot_precision_regions(X, y, classifier=ppn)
    plt.figure()
    plt.title('Decision Region')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    ###
    #Information Gain Visualization
    ###
    x = np.arange(0.0, 1.0, 0.01)
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e * 0.5 if e else None for e in ent]
    err = [error(i) for i in x]
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Missclasification Error'], ['-', '-', '--', '-.'], ['black', 'lightgray', 'red', 'green', 'cyan']):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k' , linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('impurity index')
    plt.show()


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


class AdalineGD(object):
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


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    --------------------
    eta : float.
        Learning rate (between 0.0 and 0.1)
    n : int
        Passes over the training dataset.
    shuffle : bool (default : True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    --------------------
    w : 1d-array
        Weights after fitting.
    cost : list
        Sum-of-Squares cost function value averaged over all training examples in each epoch.
    """
    def __init__(self, eta=0.1, n= 50, shuffle=True, random_state=None):
        self.eta = eta
        self.n = n
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit the training data.
        
        Parameters
        --------------------
        X : {array-like}, shape = [samples, features]
            Training vectors, where samples is the number of examples and features is the number of features.
        y : {array-like}, shape = [samples]
            Target values.
            
            Returns
            --------------------
            self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost = []
        for i in range(self.n):
            if self.shuffle:
                X, y = self._shuffle(X,y)
                cost=[]
                for xi, target in zip(X, y):
                    cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data."""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize the weights to small random numbers."""
        self.rgen = np.random.RandomState(self.random_state)
        self.w = self.rgen_normal(loc=0.0, scale=0.1, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply adaline learning rule to update the weights."""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w[1:] += self.eta * xi.dot(error)
        self.w[0] += self.eta * error
        cost = 0.5 * error ** 2 
        return cost

    def net_input(self, X):
        """Calculate the net input."""
        return np.dot(X, self.w[1:]) + self.w[0]
    
    def activation(self, X):
        """Compute linear activation."""
        return X
    
    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

def plot_precision_regions(X, y, classifier, resolution=0.02):
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

def gini(p):
    """Gini Impurity"""
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

def entropy(p):
    """Entropy Impurity"""
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)

def error(p):
    """Classification Error Impurity"""
    return 1 - np.max([p, 1 - p])



if __name__ == "__main__":
    main()
