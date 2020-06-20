from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


def main():
    ###
    #Example of scikit-learn Perceptron
    ###
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print('Class labels: ', np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    print('Labels counts in y_train: ', np.bincount(y))
    print('Labels counts in y_train: ', np.bincount(y_train))
    print('Labels counts in y_train: ', np.bincount(y_test))
    sc = StandardScaler()
    sc.fit(X_train)
    X_train__std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train__std, y_train)
    y_pred = ppn.predict(X_test_std)
    print('Misclassified examples: %d' % (y_test != y_pred).sum())
    print('Accuracy %.3f' % accuracy_score(y_test, y_pred))
    print('Accuracy %.3f' % ppn.score(X_test_std, y_test))
    X_combined_std = np.vstack((X_train__std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plt.figure()
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
    plt.xlabel( 'petal length [standardized]' )
    plt.ylabel( 'petal width [standardized]' )
    plt.legend(loc= 'upper left' )
    plt.tight_layout()
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.figure()
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='black')
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('$\phi (z)$')
    #y-axis ticks and gridline
    plt.yticks([ 0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.tight_layout()
    z = np.arange(-10, 10, 0.1)
    phi_z = sigmoid(z)
    plt.figure()
    c0 = [cost_0(x) for x in z]
    plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')
    c1 = [cost_1(x) for x in z]
    plt.plot(phi_z, c1, linestyle='--', label='J(w) if y=1')
    plt.ylim(0.0, 0.5)
    plt.xlim([0, 1])
    plt.xlabel('$\phi$(z)')
    plt.ylabel('J(w)')
    plt.legend(loc='best')
    plt.tight_layout()
    #Example of Logistic REgression model using Gradient Descent
    X_train_01__subset = X_train[(y_train == 0) | (y_train == 1)]
    y_train_01__subset = y_train[(y_train == 0) | (y_train == 1)]
    lrgd = LogisticRegressionGD(eta=0.05, n=1000)
    lrgd.fit(X_train_01__subset, y_train_01__subset)
    plt.figure()
    plot_decision_regions(X=X_train_01__subset, y=y_train_01__subset, classifier=lrgd)
    plt.xlabel( 'petal length [standardized]' )
    plt.ylabel( 'petal width [standardized]' )
    plt.legend(loc= 'upper left' )
    plt.tight_layout()
    #Example of Logistic Regression Model using scikit Learning
    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
    lr.fit(X_train__std, y_train)
    plt.figure()
    plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
    plt.xlabel( 'petal length [standardized]' )
    plt.ylabel( 'petal width [standardized]' )
    plt.legend( loc = 'upper left' )
    plt.tight_layout()
    lr.predict_proba(X_test_std[:3, :])
    #The L2 Regularization Path for the Two Weight Coefficients
    plt.figure()
    weights, params = [], []
    for c in np.arange(-5, 5):
        lr = LogisticRegression(C=10.**c, random_state=1, solver='lbfgs', multi_class='ovr')
        lr.fit(X_train__std, y_train)
        weights.append(lr.coef_[1])
        params.append(10.**c)
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='petal length')
    plt.plot(params, weights[:, 1], linestyle = '--', label='petal width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    #Example of Support Vector Machines (SVM)
    plt.figure()
    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train__std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx = range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #Example of kernel SVMs
    ppn = SGDClassifier(loss='perceptron')
    lr = SGDClassifier(loss='log')
    svm = SGDClassifier(loss='hinge')
    np.random.seed(1)
    X_xor = np.random.randn(200,2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 1)
    y_xor = np.where(y_xor, 1, -1)
    plt.figure()
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='blue', marker='x', label='1')
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='red', marker='s', label='-1')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


class LogisticRegressionGD(object):
    """Logistic Regression Classifier using Gradient Descent.
    
    Parameters
    --------------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    --------------------
    w : 1D - Array
        Weights after fitting.
    cost : list
        logistic cost function value in each epoch.
    """
    def __init__(self, eta=0.5, n=100, random_state=1):
        self.eta = eta
        self.n = n
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit the training data.
        
        Parameters
        --------------------
        X : {array-like}, shape = [samples, features]
            Training vectors, where samples is the number of examples and features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
        
        Returns
        --------------------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost= []
        for i in range(self.n):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w[1:] += self.eta *X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            #Note that we compute the logistic 'cost' now instead of the sum of squared errors cost.
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate the net input."""
        return np.dot(X, self.w[1:]) +self.w[0]
    
    def activation(self, z):
        """Compute logistic sigmoid activation."""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, 0) #Equivalent to : return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """Setup marker generator and color map"""
    from matplotlib.colors import ListedColormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],marker=markers[idx], label=cl, edgecolors='black')
    #Highlight test examples
    if test_idx:
        #plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolors='black', alpha=1.0, linewidths=1, marker='o', s=100, label='test sets')

def sigmoid(z):
    """Simple sigmoid function. think of this as an S-shaped curve"""
    return 1.0 / (1.0 + np.exp(z))

def cost_1(z):
    """Cost function for classifying the second training example"""
    return - np.log(sigmoid(z))

def cost_0(z):
    """Cost function for classifying the first training example"""
    return np.log( 1 - sigmoid(z))


if __name__ == "__main__":
    main()