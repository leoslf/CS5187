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
from sklearn.metrics import confusion_matrix, classification_report, log_loss, make_scorer

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
    from keras.preprocessing.image import * 

    from keras import backend as K
    from keras.utils import generic_utils


logger = logging.getLogger(__name__)
LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)


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
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def load_dataset(filenames):
    return list(map(scipy.io.loadmat, filenames))

def reshape_dataset(dataset, one_hot = True):
    X, Y = dataset["X"], dataset["y"]
    X = X.transpose([3, 0, 1, 2])
    return np.array(X), np.array(OneHotEncoder(sparse=False).fit_transform(Y) if one_hot else Y)

def reporting(name, y_true, y_pred, labels = None):
    report = classification_report(y_true, y_pred, digits=8, target_names=labels)
    report_filename = "%s_test_classification_report.txt" % name
    with open(report_filename, "w") as f:
        f.write(report)
    logger.info("Saved \"%s\"", report_filename)

    confusion = confusion_matrix(y_true, y_pred)
    confusion_filename = "%s_test_confusion_matrix.txt" % name
    np.savetxt(confusion_filename, confusion, delimiter=",")
    logger.info("Saved \"%s\"", confusion_filename)


def smooth_L1(sigma):
    def smooth_L1_implementation(x):
        x_abs = K.abs(x)
        abs_lt_1 = K.stop_gradient(K.less(x_abs, 1. / sigma ** 2))
        return abs_lt_1 * ((0.5 * sigma ** 2) * K.square(x)) + (1 - abs_lt_1) * (x_abs - (0.5 / sigma ** 2))
    return smooth_L1_implementation

def binary_crossentropy(p_true, p_pred):
    return - K.log(p_true * p_pred + (1 - p_true) * (1 - p_pred))

def binary_classifier_loss_regression(inside_weights, outside_weights, sigma, dim = [1]):
    smooth_l1 = smooth_L1(sigma)
    def loss(bbox_gt, bbox_pred):
        bbox_difference = bbox_pred - bbox_gt
        inside_difference = inside_weights * bbox_difference
        inside_difference_abs = K.abs(inside_difference)
        inside_loss = smooth_l1(inside_difference)
        outside_loss = outside_weights * inside_loss

        return K.mean(K.sum(outside_loss, axis = dim))

    return loss

# function to make an image montage
def image_montage(X, imsize=None, maxw=10):
    """X can be a list of images, or a matrix of vectorized images.
      Specify imsize when X is a matrix."""
    tmp = []
    numimgs = len(X)
    
    # create a list of images (reshape if necessary)
    for i in range(0, numimgs):
        if imsize != None:
            tmp.append(X[i].reshape(imsize))
        else:
            tmp.append(X[i])
    
    # add blanks
    if (numimgs > maxw) and (np.mod(numimgs, maxw) > 0):
        leftover = maxw - np.mod(numimgs, maxw)
        # meanimg = 0.5*(X[0].max()+X[0].min())
        for i in range(0,leftover):
            tmp.append(np.ones(tmp[0].shape))#  * meanimg)
    
    # make the montage
    tmp2 = []
    for i in range(0,len(tmp),maxw):
        tmp2.append(np.hstack(tmp[i:i+maxw]))
    montimg = np.vstack(tmp2) 
    return montimg

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

