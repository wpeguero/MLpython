from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
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
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print('Misclassified examples: %d' % (y_test != y_pred).sum())
    print('Accuracy %.3f' % accuracy_score(y_test, y_pred))
    print('Accuracy %.3f' % ppn.score(X_test_std, y_test))
    X_combined_std = np.vstack((X_train_std, X_test_std))
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
    plt.show()


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
    return 1.0 / (1.0 + np.exp(z))


if __name__ == "__main__":
    main()