import numpy as np
from sklearn import tree

def get_dt_model(random_state=42, max_depth=2, min_samples_leaf=5):
    clf = tree.DecisionTreeRegressor(random_state=random_state, criterion='mae', max_depth=2, min_samples_leaf=5)
    return clf

def predict_confidence(clf, X):
    preds = clf.predict(X)
    nodes = clf.apply(X)
    sigmas = clf.tree_.impurity[nodes] * np.sqrt(2)
    return np.array([preds, sigmas]).T

def get_models(num_features, random_state=42):
    clfs = []
    clfs.append(get_dt_model(random_state=random_state, max_depth=2, min_samples_leaf=5))
    return clfs
