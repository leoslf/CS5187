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

if __name__ == "__main__":
    with open("evaluation.pickle", "rb") as f:
        evaluation = pickle.load(f)

    print (evaluation)
