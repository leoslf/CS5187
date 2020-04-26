import sys
import os

import operator
import pickle

import warnings
import logging
import argparse

import numpy as np

from functools import *

from sklearn.exceptions import DataConversionWarning

from streetnumber.model import *
from streetnumber.nn_classifier import *
from streetnumber.feature_extractor import *
from streetnumber.utils import *

logger = logging.getLogger(__name__)

dataset_filenames = ["%s_32x32.mat" % name for name in ["train", "test"]]

labels = list(map(str, range(10)))

parser = argparse.ArgumentParser(description="Train and/or run models.")
parser.add_argument("--train-nn", dest="train_nn", default=False, action="store_true",
                   help="Training the neural network")
parser.add_argument("--no-train-nn", dest="train_nn", action="store_false",
                   help="Explicitly disabling the training of the neural network")
parser.add_argument("--preprocess-svm-dataset", dest="preprocess_svm_dataset", default=False, action="store_true",
                   help="Preprocess the datasets for svm with trained FeatureExtractor")
argv = parser.parse_args()

if __name__ == "__main__":
    # Supress sklearn DataConversionWarning
    warnings.filterwarnings(action="ignore", category=DataConversionWarning)

    # Log Level
    logging.basicConfig(level=logging.DEBUG)

    # numpy print options
    np.set_printoptions(precision = 2, suppress = True, threshold=sys.maxsize, linewidth=200)



    # Load Dataset
    train_onehot, test_onehot = list(map(reshape_dataset, load_dataset(dataset_filenames)))
    train, test = list(map(partial(reshape_dataset, one_hot = False), load_dataset(dataset_filenames)))

    model = NNClassifier()
    if argv.train_nn:
        history = model.fit(*train_onehot)
        with open("history.pickle", "wb") as f:
            pickle.dump(history, f)

    train_X, train_Y = train
    test_X, test_Y_onehot = test_onehot
    _, test_Y = test

    evaluation = model.evaluate(*test_onehot)
    with open("evaluation.pickle", "wb") as f:
        pickle.dump(evaluation, f)
    logger.info("Saved \"evaluation.pickle\"")

    # print (evaluation)
    predict_Y = model.predict(test_X)
    reporting("nn", np.argmax(test_Y_onehot, axis=-1), predict_Y)

    if argv.preprocess_svm_dataset:
        # Feature Extractor
        feature_extractor = FeatureExtractor()
        # Feature vector Representation
        train_embeddings = feature_extractor.predict(train_X)
        test_embeddings = feature_extractor.predict(test_X)

        svm_train = (train_embeddings, train_Y)
        svm_test = (test_embeddings, test_Y)

        with open("svm_train.pickle", "wb") as f:
            pickle.dump(svm_train, f)

        with open("svm_test.pickle", "wb") as f:
            pickle.dump(svm_test, f)

