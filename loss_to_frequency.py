import numpy as np

def process(model):
    cross_entropy = np.loadtxt("%s_predict_crossentropy.csv" % model, delimiter=",")

    rounded = np.round(cross_entropy, 2)
    unique_losses, unique_counts = np.unique(rounded, return_counts = True)

    output = np.column_stack((unique_losses, unique_counts, unique_counts / np.sum(unique_counts)))
    np.savetxt("%s_predict_crossentropy_group_by_frequency.csv" % model, output, delimiter=",")

if __name__ == "__main__":
    for model in ["svm", "nn"]:
        process(model)

