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
    return type(clf) == tf.keras.models.Model

def get_models(num_features, random_state=42):
    clfs = []
    clfs.append(GBR(**{'n_estimators': 300,
          'max_depth': 4,
          'min_samples_split': 8,
          'learning_rate': 0.0075,
          'loss': 'huber',
          'random_state': random_state}))
    clfs.append(lm.Ridge(alpha=6, random_state=random_state))
    clfs.append(lm.SGDRegressor(loss='huber', alpha=0.5, random_state=random_state))
    clfs.append(lm.HuberRegressor(epsilon=1, alpha=0.1, max_iter=10000))
    clfs.append(get_nn_model(num_features=num_features, random_state=random_state))
    clfs.append(svm.SVR(kernel='linear', C=0.1))
    return clfs