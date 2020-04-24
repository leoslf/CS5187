import sys
import os

import operator

from functools import *

import logging

import json

import numpy as np

import warnings

import scipy
import scipy.io
import scipy.stats

import pickle

from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import multilabel_confusion_matrix

from pprint import pprint

from streetnumber.model import *
from streetnumber.feature_extractor import *
from streetnumber.utils import *

logger = logging.getLogger(__name__)

dataset_filenames = ["%s_32x32.mat" % name for name in ["train", "test"]]

if __name__ == "__main__":
    # Supress sklearn DataConversionWarning
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # Log Level
    logging.basicConfig(level=logging.DEBUG)

    # numpy print options
    np.set_printoptions(precision = 2, suppress = True, threshold=sys.maxsize, linewidth=200)
    
    train_onehot, test_onehot = list(map(reshape_dataset, load_dataset(dataset_filenames)))
    train, test = list(map(partial(reshape_dataset, one_hot = False), load_dataset(dataset_filenames)))

    model = FeatureExtractor()
    # history = model.fit(*train_onehot)
    # with open("history.pickle", "wb") as f:
    #     pickle.dump(history, f)

    evaluation = model.evaluate(*test_onehot)
    with open("evaluation.pickle", "wb") as f:
        pickle.dump(evaluation, f)

    print (evaluation)
    





    


    








