import numpy as np
from scipy import stats
from collections import Counter
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)

    def leaf(self, x_test):

        if x_test[self.col] <= self.split:
            return self.lchild.leaf(x_test)
        else:
            return self.rchild.leaf(x_test)



class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        return self.prediction

    def leaf(self, x_test):
        return self


class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None, max_features=10):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini
        self.max_features = max_features

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """

        #Small leaf
        if X.shape[0]<=self.min_samples_leaf:
            #print("creating leaf...")
            return self.create_leaf(y)

        #print("max features",self.max_features )
        col,split = rf_bestsplit(X,y,self.loss, self.max_features,self.min_samples_leaf)
        #print("col split from fit_", col, split)
        # No better splits
        if col==-1:
            return self.create_leaf(y)

        lchild = self.fit_(X[X[:,col]<=split,:], y[X[:,col]<=split])
        rchild = self.fit_(X[X[:,col]>split,:], y[X[:,col]>split])

        return DecisionNode(col, split,lchild,rchild)




    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        #list_predictions=self.root.predict(X_test)
        #return list_predictions
        n = X_test.shape[0]
        y_pred=[]

        for i in range(0,n):
            value_pred=self.root.predict(X_test[i,:])
            y_pred.append(value_pred)

        return np.array(y_pred)


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1,max_features=10):
        super().__init__(min_samples_leaf, loss=np.std, max_features=max_features)
    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_pred=self.predict(X_test)
        return r2_score(y_test, y_pred)

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y,np.mean(y))


class ClassifierTree621(DecisionTree621):

    def __init__(self, min_samples_leaf=1, max_features=10):
        super().__init__(min_samples_leaf, loss=gini, max_features=max_features)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        mode = Counter(y).most_common(1)
        return LeafNode(y,mode[0][0])

def gini(y):
    "Return the gini impurity score for values in y"
    _, counts = np.unique(y, return_counts=True)
    n = len(y)
    return 1 - np.sum( (counts / n)**2 )

def rf_bestsplit(X, y, loss, max_features, min_samples_leaf):

    best_col = -1
    best_split = -1
    best_loss=loss(y)
    p = X.shape[1]
    n = X.shape[0]
    len_y=y.shape[0]

    # Picking max obs (rows)
    k=11
    if n<11:
        k=n

    # Picking max features (cols):
    list_features = np.random.choice(range(0,p),size=int(round(max_features*p,0)),replace=False)

    for col in list_features:
        list_candidates=np.random.choice(X[:,col],size=k,replace=False)

        for split in list_candidates:

            yl=y[X[:,col]<=split]
            yr=y[X[:,col]>split]
            len_yl=yl.shape[0]
            len_yr=yr.shape[0]

            if (len_yl<min_samples_leaf) or (len_yr<min_samples_leaf):
                continue

            l = (len_yl * loss(yl) + len_yr * loss(yr)) / len_y

            if l==0:
                return col,split

            if l < best_loss:
                best_col=col
                best_split=split
                best_loss=l

    return best_col,best_split
