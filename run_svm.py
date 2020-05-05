import os
import numpy as np

import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# import logging
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

    # predict_Y = clf.predict_proba(test_embeddings)
    # # reporting("svm", test_Y % 10, predict_Y % 10) # , list(map(str, list(range(1, 10)) + [0])))
    # np.savetxt("svm_predict_Y.txt", predict_Y)

    support_vectors = clf.best_estimator_.support_vectors_
    np.savetxt("support_vectors.txt", support_vectors)

    support = clf.best_estimator_.support_
    np.savetxt("support.txt", support)

    dataset_filenames = ["%s_32x32.mat" % name for name in ["train", "test"]]
    train, test = list(map(partial(reshape_dataset, one_hot = False), load_dataset(dataset_filenames)))

    trainX, trainY = train

    import imageio
    montage = image_montage(trainX[support], imsize = (32, 32, 3), maxw=100)
    imageio.imwrite("montage_sv.png", montage)

    predicted_train_Y = clf.predict(train_embeddings)
    misclassified_index = predicted_train_Y != (train_Y % 10)
    np.savetxt("misclassified_index_svm.txt", misclassified_index)

    montage = image_montage(trainX[misclassified_index], imsize = (32, 32, 3), maxw=100)
    import imageio
    imageio.imwrite("montage_svm.png", montage)

    misclassified_index_nn = np.loadtxt("misclassified_index_nn.txt").astype(bool)
    
    both_wrong = misclassified_index_nn & misclassified_index
    montage_both_wrong = image_montage(trainX[both_wrong], imsize = (32, 32, 3), maxw=100)
    imageio.imwrite("montage_both_wrong.png", montage_both_wrong)

    svm_wrong = misclassified_index & (~both_wrong)
    montage_svm_wrong = image_montage(trainX[svm_wrong], imsize = (32, 32, 3), maxw=100)
    imageio.imwrite("montage_svm_wrong.png", montage_svm_wrong)

    nn_wrong = misclassified_index_nn & (~both_wrong)
    montage_nn_wrong = image_montage(trainX[nn_wrong], imsize = (32, 32, 3), maxw=100)
    imageio.imwrite("montage_nn_wrong.png", montage_nn_wrong)




