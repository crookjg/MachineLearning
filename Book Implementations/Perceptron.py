import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

s = os.path.join('https://archive.ics.uci.edu', 'ml',
                    'machine-learning-databases', 'iris',
                    'iris.data')
df = pd.read_csv(s, header=None, encoding='utf-8')
df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 1, -1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

def plot_data():
    global X
    # plot data
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')
    plt.xlabel('Sepal Length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

def new_perceptron():
    global X
    global y
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1),
            ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('# of updates')
    plt.show()

class Perceptron(object):
    """     Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset
    random_state : int
        Random number generator seed for random weight initialization
    
    Attributes
    ------------
    w_ : 1D array
        Weights after fitting
    errors_ : list
        Number of misclassifications (updates) in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """     Fit training data

        Parameters
        ------------
        X : array-like, shape = [n_examples, n_features]
            Training vectors, where n_examples is the # of examples
            and n_features is the # of features
        y : array-like, shape = [n_examples]
            Target values

        Returns
        -----------
        self : object
        """

        rgen = np.random.RandomState(self.random_state)
        """
            Initialize the weights in self.w_ to a vector R^(m+1) where 
            m = # of dimensions (features) in the dataset and we add the
            1 for the first element in this vector that represents the
            bias unit. self.w_[0] represents the bias unit.

            Do not initialize the weights to 0 because the learning rate, eta,
            only has an effect on the classification outcome if the weights are
            non-zero values.
        """
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.errors_ = []

        """
            Loop through all individual examples in the training dataset and
            update the weights according to the percetron learning rule.
        """
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """     Calculate net input """
        #   The below list comprehension is the classic python code to return the
        #   dot product instead of using NumPy
        # return sum([i * j for i, j in zip(X, self.w_[1:] + self.w_[0])])
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """     Return class label after unit step """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

new_perceptron()