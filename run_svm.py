import os
import numpy as np

import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from streetnumber.utils import *

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

param_grid = {
    "kernel": ["rbf", "linear"],
    # "gamma": [1e-3, 1e-4],
    "C": [1, 10, 100, 1000]
}

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    svm_train = load_pickle("svm_train.pickle")
    svm_test = load_pickle("svm_test.pickle")

    train_embeddings, train_Y = svm_train
    test_embeddings, test_Y = svm_test

    # Flattening Y for sklearn API
    train_Y = train_Y.flatten()
    test_Y = test_Y.flatten()

    # clf = SVC(kernel = "linear")
    # logger.info("training default linear SVC")
    # clf.fit(train_embeddings, train_Y)
    # logger.info("finished training default linear SVC")
    # predict_Y = clf.predict(test_embeddings)
    # reporting("svm_default", test_Y, predict_Y)

    if os.path.exists("svm.pickle"):
        with open("svm.pickle", "rb") as f:
            clf = pickle.load(f)
    else:
        # SVM
        clf = GridSearchCV(SVC(probability=True), param_grid, scoring = LogLoss, verbose=10, n_jobs=None) # categorical_crossentropy
        logger.info("training SVC")
        clf.fit(train_embeddings, train_Y % 10)
        logger.info("finished training SVC")
        
        # Pickling SVM Model
        with open("svm.pickle", "wb") as f:
            pickle.dump(clf, f)
        logger.info("Saved \"svm.pickle\"")

    predict_Y = clf.predict(test_embeddings)
    reporting("svm", test_Y % 10, predict_Y % 10) # , list(map(str, list(range(1, 10)) + [0])))

