import numpy as np
import sklearn.linear_model as lm

min_sigma = 70
max_delta = 1000
eps = np.finfo(np.float32).eps

def _laplace(delta, sigma):
    return (delta / sigma) * np.sqrt(2) + np.log(sigma * np.sqrt(2))

def laplace(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, min_sigma)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), max_delta)
    metric = _laplace(delta, sigma_clip)
    return np.mean(metric)

def laplace_optimal(fvc_mae):
    deltas = np.minimum(fvc_mae, max_delta)
    sigmas = np.maximum(deltas * np.sqrt(2), min_sigma)
    return np.mean(_laplace(deltas, sigmas))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mare(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (y_true + eps))

def get_coef(x, y): 
    errors = []
    clfs = []
    for i in range(len(x)):
        x_fold = np.concatenate([x[: i], x[i+1: ]])
        y_fold = np.concatenate([y[: i], y[i+1: ]])
        clf = lm.LinearRegression()
        clf.fit(x_fold.reshape(-1, 1), y_fold)
        errors.append(mae(clf.predict(x_fold.reshape(-1, 1)), y_fold))
        clfs.append(clf)
        
    i = np.argmin(errors)
    clf = clfs[i]
    return clf.coef_