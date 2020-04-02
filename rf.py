import numpy as np
from sklearn.utils import resample
from dtree import *
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from random import sample

#cd /Users/ivettesulca/Desktop/Intro_Machine_Learning/projects/rf-ivesulca
class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.oob_idxs = []

    def fit(self, X, y):

        list_trees = []
        orig_idxs = np.array(range(0,X.shape[0]))
        self.oob_idxs = []

        for i in range(0,self.n_estimators):

            X_n,y_n,idx_inbag = self.bootstrap(X, y, X.shape[0])

            if isinstance(self, RandomForestRegressor621):
                model=RegressionTree621(min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
            else:
                model=ClassifierTree621(min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)

            model.fit(X_n,y_n)
            list_trees.append(model.root)

            #Only if OOBScore activated, store oob idxs
            if self.oob_score == True:
                self.oob_idxs.append(np.delete(orig_idxs,idx_inbag))

        self.trees = list_trees
        self.nunique = len(np.unique(y))


        #Calculating OOB Scores
        if self.oob_score == True:
            self.oob_score_=self.compute_oob_score(X,y)


    def bootstrap(self, X, y, size_x):

        n=int(round((2/3)*size_x,0))
        idx = sample(range(0,size_x), n)
        X_n = X[idx]
        y_n = y[idx]

        return X_n,y_n,idx

class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = np.array([])
        self.min_samples_leaf = min_samples_leaf
        self.max_features=max_features

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        list_pred=[]
        for obs in X_test:
            #print("********obs:",obs)
            pred=self.predict_obs(obs)
            #print("final pred",pred)
            list_pred.append(pred)

        return  np.array(list_pred)

    def predict_obs(self, x_test):

        nobs=0
        ysum=0
        for t in self.trees:
            leaf = t.leaf(x_test)
            nobs = nobs+leaf.n
            ysum = ysum+ leaf.predict(x_test)*leaf.n

        return ysum/nobs


    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)


    def compute_oob_score(self, X, y):
        oob_counts = np.zeros(X.shape[0])
        oob_preds = np.zeros(X.shape[0])

        i_tree=0
        for t in self.trees:
            list_oob = self.oob_idxs[i_tree]
            for idx in list_oob:
                leaf = t.leaf(X[idx,:])
                oob_preds[idx] +=  leaf.predict(X[idx,:]) * leaf.n
                oob_counts[idx] += leaf.n
            i_tree+=1

        oob_avg_preds = np.divide(oob_preds[oob_counts>0],oob_counts[oob_counts>0])

        oob_r2_score= r2_score(y[oob_counts>0], oob_avg_preds)
        return oob_r2_score

class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = np.array([])
        self.min_samples_leaf = min_samples_leaf
        self.max_features=max_features

    def predict(self, X_test) -> np.ndarray:
        list_pred=[]
        for obs in X_test:
            #print("********obs:",obs)
            pred=self.predict_obs(obs)
            #print("final pred",pred)
            list_pred.append(pred)

        return  np.array(list_pred)

    def predict_obs(self, x_test):
        #n_classes = self.nunique
        class_count=np.zeros(self.nunique+1)

        for t in self.trees:
            leaf = t.leaf(x_test)
            pred = leaf.predict(x_test)
            class_count[pred]+=leaf.n

        return np.argmax(class_count)


    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def compute_oob_score(self, X, y):
        oob_counts = np.zeros(X.shape[0])

        #Create 2d array: rows -> observations, cols -> unique classes
        oob_preds = np.zeros((X.shape[0], self.nunique))

        i_tree=0
        for t in self.trees:
            list_oob = self.oob_idxs[i_tree]
            for idx in list_oob:
                leaf = t.leaf(X[idx,:])
                class_pred = leaf.predict(X[idx,:])
                oob_preds[idx,class_pred] += leaf.n
                oob_counts[idx] +=1
            i_tree+=1

        oob_votes = []

        for i in oob_preds[oob_counts>0,:]:
            oob_votes.append(np.argmax(i))

        oob_r2_score= r2_score(y[oob_counts>0], oob_votes)

        return oob_r2_score
