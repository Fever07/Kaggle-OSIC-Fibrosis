import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import sklearn.linear_model as lm
import numpy as np
from lightgbm import LGBMRegressor as lgmbr

C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = tf.abs(y_pred[:, 1] - y_pred[:, 0])
    fvc_pred = y_pred[:, 0]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)

def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def mloss2(y_true, y_pred):
    alpha = 1.0
    # return alpha * K.mean(tf.abs(y_true - y_pred[:, 0])) + (1.0 - alpha) * score(y_true, y_pred)
    return K.mean(tf.abs(y_true - y_pred[:, 0]))

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
    return loss

def get_nn_model(num_features=9, random_state=42):
    np.random.seed(random_state)
    tf.compat.v1.set_random_seed(random_state)
    inp = keras.layers.Input(shape=(num_features, ))
    out = inp
    out = keras.layers.Dense(128, activation='relu')(out)
    out = keras.layers.Dense(64, activation='relu')(out)
    out = keras.layers.Dense(32, activation='relu')(out)
    p1 = keras.layers.Dense(3, activation="linear", name="p1")(out)
    p2 = keras.layers.Dense(3, activation="relu", name="p2")(out)
    p3 = keras.layers.Dense(3, activation="sigmoid", name="p3")(out)
    out = keras.layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1) + tf.cumsum(x[2], axis=1), name="preds")([p1, p2, p3])
    out = keras.layers.Dense(1, activation='linear')(out)
    
    model = keras.models.Model(inp, out)
    model.compile(loss='MAE',
                 optimizer=keras.optimizers.Adam(),
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
    clfs.append(lm.HuberRegressor(max_iter=10000))
    if include_nn:
        clfs.append(get_nn_model(num_features=num_features, random_state=random_state))
    clfs.append(svm.SVR(kernel='linear'))
    # clfs.append(lgmbr(
    #     max_depth=4,
    #     n_estimators=64,
    #     num_leaves=4,
    #     random_state=random_state,
    #     min_child_samples=32,
    # ))
    return clfs