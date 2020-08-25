import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import sklearn.linear_model as lm
import numpy as np

def get_nn_model(num_features=9, random_state=42):
    np.random.seed(random_state)
    tf.compat.v1.set_random_seed(random_state)
    inp = keras.layers.Input(shape=(num_features, ))
    out = inp
    out = keras.layers.Dense(96, activation='relu')(out)
    out = keras.layers.Dense(1, activation='linear')(out)
    
    model = keras.models.Model(inp, out)
    model.compile(loss='MAE',
                 optimizer='RMSProp',
                 metrics=['MAE'])
    return model

def is_net(clf):
    if tf.__version__[0] == '1':
        return type(clf) == tf.keras.models.Model
    else:
        return type(clf) == tf.python.keras.engine.functional.Functional

def get_models(num_features, include_nn=True, random_state=42):
    clfs = []
    # clfs.append(GBR(**{'n_estimators': 300,
    #       'max_depth': 4,
    #       'min_samples_split': 8,
    #       'learning_rate': 0.0075,
    #       'loss': 'huber',
    #       'random_state': random_state}))
    clfs.append(lm.Ridge(random_state=random_state))
    # clfs.append(lm.SGDRegressor(loss='huber', random_state=random_state))
    clfs.append(lm.HuberRegressor(max_iter=10000))
    if include_nn:
        clfs.append(get_nn_model(num_features=num_features, random_state=random_state))
    # clfs.append(svm.SVR(kernel='linear'))
    return clfs