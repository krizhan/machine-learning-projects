import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def create_permutation(length):
    return np.random.permutation(length)


def shuffle_data(data, permutation):
    # We shuffle the matrix with the permutation
    return data[permutation]


def split_data(data, num_folds, fold):
    # We split the data into equal partitions
    partitioned_data = np.split(data, num_folds)
    data_fold_matrix, data_rest_matrix = None, None
    for partition in range(num_folds):
        if partition == fold:
            data_fold_matrix = partitioned_data[partition]
        else:
            data_rest_matrix = partitioned_data[partition] if data_rest_matrix is None \
                else np.concatenate((data_rest_matrix, partitioned_data[partition]))
    return data_fold_matrix, data_rest_matrix


def train_model(data_x_matrix, data_y_matrix, lambd):
    # Extract the X and Y Variables
    data_x_transpose = np.transpose(data_x_matrix)
    # Perform the calculations
    xt_times_x, xt_times_y = np.dot(data_x_transpose, data_x_matrix), np.dot(data_x_transpose, data_y_matrix)
    lambda_identity = lambd * np.identity(len(data_x_matrix[0, :]))
    inverse_matrix = np.linalg.inv(xt_times_x + lambda_identity)
    return np.dot(inverse_matrix, xt_times_y)


def predict(data_x_matrix, model):
    # Extract the X variables and multiple by the model parameters
    return np.dot(data_x_matrix, model)


def loss(data_x_matrix, data_y_matrix, model):
    predictions = predict(data_x_matrix, model)
    difference = np.subtract(predictions, data_y_matrix)
    return np.sum(np.square(difference)) / len(data_y_matrix)


def cross_validation(data_x_matrix, data_y_matrix, num_folds, lambd_seq):
    assert len(data_x_matrix) == len(data_y_matrix)
    data_perm = create_permutation(len(data_test_X))
    data_x_matrix = shuffle_data(data_x_matrix, data_perm)
    data_y_matrix = shuffle_data(data_y_matrix, data_perm)

    cv_error = []
    for each_lambda in lambd_seq:
        cv_loss_lmd = 0.
        for fold in range(num_folds):
            val_cv_x, train_cv_x = split_data(data_x_matrix, num_folds, fold)
            assert len(train_cv_x) == (num_folds - 1) * len(val_cv_x)

            val_cv_y, train_cv_y = split_data(data_y_matrix, num_folds, fold)
            assert len(train_cv_y) == (num_folds - 1) * len(val_cv_y)

            each_model = train_model(train_cv_x, train_cv_y, each_lambda)
            cv_loss_lmd += loss(val_cv_x, val_cv_y, each_model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error


if __name__ == "__main__":

    dataset = sio.loadmat("./dataset.mat")

    data_train_X = dataset["data_train_X"]
    data_train_y = dataset["data_train_y"][0]

    data_test_X = dataset["data_test_X"]
    data_test_y = dataset["data_test_y"][0]

    lambda_seq = np.arange(0.02, 1.515, 0.03)
    assert len(data_train_X) == len(data_train_y)
    data_train_perm = create_permutation(len(data_train_X))

    data_train_X = shuffle_data(data_train_X, data_train_perm)
    data_train_y = shuffle_data(data_train_y, data_train_perm)

    testing_errors = []
    training_errors = []
    for lamb in lambda_seq:
        model = train_model(data_train_X, data_train_y, lamb)
        testing_errors.append(loss(data_test_X, data_test_y, model))
        training_errors.append(loss(data_train_X, data_train_y, model))
    five_fold_cv_error = cross_validation(data_train_X, data_train_y, 5, lambda_seq)
    ten_fold_cv_error = cross_validation(data_train_X, data_train_y, 10, lambda_seq)

    assert len(lambda_seq) == len(testing_errors)
    assert len(lambda_seq) == len(training_errors)
    assert len(lambda_seq) == len(five_fold_cv_error)
    assert len(lambda_seq) == len(ten_fold_cv_error)

    plt.plot(lambda_seq, training_errors, marker=".")
    plt.plot(lambda_seq, testing_errors, marker=".")
    plt.plot(lambda_seq, five_fold_cv_error, marker=".")
    plt.plot(lambda_seq, ten_fold_cv_error, marker=".")
    plt.legend(['Training Errors', 'Testing Errors', '5-Fold Cross Validation', '10-Fold Cross Validation'], loc='upper right')
    plt.ylabel('Average Squared Error Loss')
    plt.xlabel('Lambda Values')
    plt.title("Lambda vs Error Loss")
    plt.savefig('RidgeRegressionPlot.png')
    plt.show()
