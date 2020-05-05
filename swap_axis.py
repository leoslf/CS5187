import numpy as np

def rotate_axis(array):
    correct_order = [9] + list(range(9))
    return array[:, correct_order]

if __name__ == "__main__":
    predict_Y = np.loadtxt("svm_predict_Y.txt")
    test_Y = np.loadtxt("nn_test_Y.txt")

    # predict_Y = rotate_axis(predict_Y)
    test_Y = rotate_axis(test_Y)
    np.savetxt("svm_predict_Y_rotated.csv", predict_Y, delimiter=",")
    np.savetxt("svm_test_Y_rotated.csv", test_Y, delimiter=",")

    np.set_printoptions(suppress=True, precision=8)
    print (test_Y)
    print (predict_Y)

    cross_entropy = - np.sum(test_Y * np.log(predict_Y), axis = -1)
    print (cross_entropy)
    np.savetxt("svm_predict_crossentropy.csv", cross_entropy, delimiter=",", fmt="%.18f")


