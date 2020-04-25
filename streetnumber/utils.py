import os
from os.path import dirname, realpath

import functools
from functools import partial, reduce

import configparser
import json

import itertools
import operator

import logging

import numpy as np

import scipy
import scipy.io

from sklearn.preprocessing import OneHotEncoder

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import tensorflow as tf

    from keras.models import *
    from keras.layers import *
    from keras.initializers import *
    from keras.optimizers import *
    from keras.regularizers import *
    from keras.objectives import *
    from keras.callbacks import * 
    from keras.losses import * 

    from keras import backend as K
    from keras.utils import generic_utils


logger = logging.getLogger(__name__)

def vectorize(f):
    @functools.wraps(f)
    def wraps(self, X, *argv, **kwargs):
        return np.array(list(map(partial(f, self, *argv, **kwargs), X)))
        # return np.apply_along_axis(partial(f, self), np.arange(len(X.shape))[1:], X, *argv, **kwargs)
    return wraps

def log(f):
    @functools.wraps(f)
    def wraps(self, *argv, **kwargs):
        logger.debug(f.__name__)
        return f(self, *argv, **kwargs)
    return wraps

def compose(*functions):
    r""" Function Composition: :math:`(f_1 \circ \cdots \circ f_n)(x)` """
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

def load_dataset(filenames):
    return list(map(scipy.io.loadmat, filenames))

def reshape_dataset(dataset, one_hot = True):
    X, Y = dataset["X"], dataset["y"]
    X = X.transpose([3, 0, 1, 2])
    return np.array(X), np.array(OneHotEncoder(sparse=False).fit_transform(Y) if one_hot else Y)


# def TP(y_true, y_pred):
#     return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
# 
# # Metrics
# def recall(y_true, y_pred):
#     positives = K.sum(K.clip(y_true, 0, 1))
#     return TP(y_true, y_pred) / (positives + K.epsilon())
# 
# def precision(y_true, y_pred):
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     return TP(y_true, y_pred) / (predicted_positives + K.epsilon())
# 
# def F1(y_true, y_pred):
#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     return 2 * (p * r) / (p + r + K.epsilon())

