import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
from contextlib import contextmanager
import re
from pprint import pprint

@contextmanager
def pushd(new_dir):
    """
    A context manager that implements the `pushd` command, letting you run a
    block of commands while in a different directory
    """
    old_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield old_dir
    finally:
        os.chdir(old_dir)

def confusion(cm, classes):
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.xlabel("prediction")
    plt.ylabel("true")
    for i, _ in enumerate(classes):
        for j, _ in enumerate(classes):
            plt.text(j, i, "%d" % (cm[i, j]), horizontalalignment = "center")

    # plt.show()

class ClassificationReportRow(object):
    columns = ["precision", "recall", "f1_score", "support"]
    def __init__(self, row):
        row_data = re.split(" +", row) # row.split("      ")
        pprint(row_data)
        self.data = {}
        # empty string
        row_data.pop(0)

        self["class"] = row_data.pop(0)
        for key, value in zip(self.columns, row_data):
            self[key] = float(value)

    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __getitem__(self, key):
        return self.data[key]


class ClassificationReport:
    def __init__(self, report):
        self.rows = []
        
        lines = report.split("\n")
        pprint (lines)
        for line in lines[2:-4]:
            if len(line) > 0:
                self.rows.append(ClassificationReportRow(line).data)
    
    @property
    def df(self):
        return pd.DataFrame.from_dict(self.rows)

def save_cm(model):
    filename = "%s_test_confusion_matrix.txt" % model
    cm = np.loadtxt(filename, delimiter = ",")
    confusion(cm, list(map(str, range(10))))
    basename, ext = os.path.splitext(filename)

    plt.title("%s Confusion Matrix" % model.upper())

    with pushd("figures"):
        plt.savefig("%s.png" % basename)

    plt.close()

def save_cr(model):
    basename = "%s_test_classification_report" % model
    with open("%s.txt" % basename) as f:
        report = ClassificationReport(f.read())
        report.df.to_csv("%s.csv" % basename, index = False)

if __name__ == "__main__":

    for model in ["nn", "svm"]:
        save_cm(model)
        save_cr(model)
