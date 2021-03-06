from DataTransform2 import *
from scipy import stats
# from sklearn import model_selection
# from matplotlib.pyplot import figure, plot, subplot, title, show, bar
# import numpy as np
# import neurolab as nl
#
# # Normalize data
# # X = stats.zscore(X);
#
# ## Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
# # Y = stats.zscore(X,0);
# # U,S,V = np.linalg.svd(Y,full_matrices=False)
# # V = V.T
# ##Components to be included as features
# # k_pca = 3
# # X = X @ V[:,0:k_pca]
# # N, M = X.shape
#
#
# # Parameters for neural network classifier
# n_hidden_units = 10  # number of hidden units
# n_train = 2  # number of networks trained in each k-fold
# learning_goal = 100  # stop criterion 1 (train mse to be reached)
# max_epochs = 64  # stop criterion 2 (max epochs in training)
# show_error_freq = 5  # frequency of training status updates
#
# # K-fold crossvalidation
# K = 3  # only five folds to speed up this example
# CV = model_selection.KFold(K, shuffle=True)
#
# # Variable for classification error
# errors = np.zeros(K)
# error_hist = np.zeros((max_epochs, K))
# bestnet = list()
# k = 0
# for train_index, test_index in CV.split(X, y):
#     print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))
#
#     # extract training and test set for current CV fold
#     X_train = X[train_index, :]
#     y_train = y[train_index]
#     X_test = X[test_index, :]
#     y_test = y[test_index]
#
#     best_train_error = 1e100
#     for i in range(n_train):
#         print('Training network {0}/{1}...'.format(i + 1, n_train))
#         # Create randomly initialized network with 2 layers
#         ann = nl.net.newff([[-3, 3]] * M, [n_hidden_units, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
#         if i == 0:
#             bestnet.append(ann)
#         # train network
#         train_error = ann.train(X_train, y_train.reshape(-1, 1), goal=learning_goal, epochs=max_epochs,
#                                 show=show_error_freq)
#         if train_error[-1] < best_train_error:
#             bestnet[k] = ann
#             best_train_error = train_error[-1]
#             error_hist[range(len(train_error)), k] = train_error
#
#     print('Best train error: {0}...'.format(best_train_error))
#     y_est = bestnet[k].sim(X_test).squeeze()
#     errors[k] = np.power(y_est - y_test, 2).sum().astype(float) / y_test.shape[0]
#     k += 1
#
# # Print the average least squares error
# print('Mean-square error: {0}'.format(np.mean(errors)))
#
# figure(figsize=(6, 7))
# subplot(2, 1, 1); bar(range(0, K), errors); title('Mean-square errors');
# subplot(2, 1, 2); plot(error_hist); title('Training error as function of BP iterations');
# figure(figsize=(6, 7));
# subplot(2, 1, 1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y');
# subplot(2, 1, 2); plot((y_est - y_test)); title('Last CV-fold: prediction error (est_y-test_y)');
# show()

# exercise 8.2.6

from matplotlib.pyplot import figure, plot, subplot, title, show, bar
import numpy as np
import neurolab as nl
from sklearn import model_selection
from sklearn.preprocessing import normalize

# y = main.data.MonthlyIncome.values
# X = main.data.ix[:, main.data.columns != "MonthlyIncome"]
#
# attributeNames = X.columns
# X = X.values

#N = len(y)
#M = len(X)

# Load Matlab data file and extract variables of interest

X = X / np.linalg.norm(X)
N, M = X.shape
C = 2


# Parameters for neural network classifier
n_hidden_units = 5  # number of hidden units
n_train = 3  # number of networks trained in each k-fold
learning_goal = 100  # stop criterion 1 (train mse to be reached)
max_epochs = 64  # stop criterion 2 (max epochs in training)
show_error_freq = 5  # frequency of training status updates

# K-fold crossvalidation
K = 5  # only five folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Variable for classification error
errors = np.zeros(K)
error_hist = np.zeros((max_epochs, K))
bestnet = list()
k = 0
for train_index, test_index in CV.split(X, y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    best_train_error = 1e100
    for i in range(n_train):
        print('Training network {0}/{1}...'.format(i + 1, n_train))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]] * M, [n_hidden_units, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
        if i == 0:
            bestnet.append(ann)
        # train network
        train_error = ann.train(X_train, y_train.reshape(-1, 1), goal=learning_goal, epochs=max_epochs,
                                show=show_error_freq)
        if train_error[-1] < best_train_error:
            bestnet[k] = ann
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)), k] = train_error

    print('Best train error: {0}...'.format(best_train_error))
    y_est = bestnet[k].sim(X_test).squeeze()
    errors[k] = np.power(y_est - y_test, 2).sum().astype(float) / y_test.shape[0]
    k += 1

# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(errors)))

figure(figsize=(6, 7));
# legend(['y_est', 'y_test'])
subplot(2, 1, 1);
bar(range(0, K), errors);
title('Mean-square errors');
subplot(2, 1, 2);
plot(error_hist);
title('Training error as function of BP iterations');
figure(figsize=(6, 7));
subplot(2, 1, 1);
plot(y_est);
plot(y_test);
title('Last CV-fold: est_y vs. test_y');
subplot(2, 1, 2);
plot((y_est - y_test));
title('Last CV-fold: prediction error (est_y-test_y)');
show()