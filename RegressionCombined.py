from DataTransform2 import *
from matplotlib.pyplot import figure, legend, plot, bar, subplot, title, xlabel, ylabel, show, clim, xticks
import sklearn.linear_model as lm
import neurolab as nl
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

X = X / np.linalg.norm(X)
avg_guess = 0
for i in range(len(y)):
    avg_guess += y[i]
avg_guess = avg_guess / len(y)
N, M = X.shape
C = 2

print('The average salery is ', avg_guess)

# Parameters for neural network classifier
n_hidden_units = np.arange(2,5,1)  # number of hidden units
n_train = 2  # number of networks trained in each k-fold
learning_goal = 0.01  # stop criterion 1 (train mse to be reached)
max_epochs = 64  # stop criterion 2 (max epochs in training)
show_error_freq = 5  # frequency of training status updates

y_est_forward = None


K = 5
CV = model_selection.KFold(K, shuffle=True)

# Variable for ann
errors_ann = np.zeros(K)
error_hist_ann = np.zeros((max_epochs,K))
bestnet = list()
errors_with_hidden = np.zeros((len(n_hidden_units), K))
k=0
hidden_layers = np.zeros(K)

Features = np.zeros((M, K))
Error_train_fs = np.empty((K, 1))
Error_test_fs = np.empty((K, 1))
Error_avg = np.empty((K,1))

for train_index, test_index in CV.split(X, y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    internal_cross_validation = 10
    best_train_error = 1e100


    for i, t in enumerate(n_hidden_units):
        print('Training with hidden units {0}/{1}...'.format(i + 1, n_hidden_units[len(n_hidden_units) - 1]))
        ann = nl.net.newff([[-3, 3]] * M, [t, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
        if i == 0:
            bestnet.append(ann)
        # train network
        train_error = ann.train(X_train, y_train.reshape(-1, 1), goal=learning_goal, epochs=max_epochs,
                                show=show_error_freq)
        errors_with_hidden[i][k - 1] = train_error[-1]
        if train_error[-1] < best_train_error:
            hidden_layers[k] = t
            bestnet[k] = ann
            best_train_error = train_error[-1]
            error_hist_ann[range(len(train_error)), k] = train_error

    print('Best train error: {0}...'.format(best_train_error))
    y_est = bestnet[k].sim(X_test).squeeze()
    errors_ann[k] = np.power(y_est - y_test, 2).sum().astype(float) / y_test.shape[0]





    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)

    textout=''
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,
                                                                          display=textout)

    Features[selected_features, k] = 1
    # .. alternatively you could use module sklearn.feature_selection

    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
    else:
        m = lm.LinearRegression(fit_intercept=True, normalize=True).fit(X_train[:, selected_features], y_train)
        Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, selected_features])).sum() / y_train.shape[0]
        Error_test_fs[k] = np.square(y_test - m.predict(X_test[:, selected_features])).sum() / y_test.shape[0]
        y_est_for = m.predict(X_test[:, selected_features])

        avg = np.empty(len(y_test))
        avg.fill(avg_guess)
        Error_avg[k] = np.square(y_test - avg).sum() / y_test.shape[0]

        figure(k)
        subplot(1, 2, 1)
        plot(range(1, len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')

        subplot(1, 3, 3)
        bmplot(attributeNames, range(1, features_record.shape[1]), -features_record[:, 1:])
        clim(-1.5, 0)
        xlabel('Iteration')

    k += 1

best_model = bestnet[errors_ann.argmin()]
best_hidden = hidden_layers[errors_ann.argmin()]
y_est_ann_outer = best_model.sim(X).squeeze()

print('Mean-square error: {0}'.format(np.mean(errors_ann)))


figure(figsize=(6,7));
subplot(2,1,1); bar(range(0,K),errors_ann); title('Mean-square errors');
subplot(2,1,2); plot(error_hist_ann); title('Training error as function of BP iterations');
figure(figsize=(6,7));
subplot(2,1,1); plot(y_est_for); plot(y_test); title('Last CV-fold in ANN: est_y vs. test_y');
subplot(2,1,2); plot(y_est_for.reshape(-1,1) - y_test.reshape(-1,1)); title('Last CV-fold: prediction error (est_y-test_y)');

show()
# DISPLAY RESULTS
figure(k)
title('Features selected by crossvalidation')
subplot(1, 3, 2)
bmplot(attributeNames, range(1, Features.shape[1] + 1), -Features)
clim(-1.5, 0)
xlabel('Crossvalidation fold')
ylabel('Attribute')

f = 2  # cross-validation fold to inspect
ff = Features[:, f - 1].nonzero()[0]
if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
else:
    m = lm.LinearRegression(fit_intercept=True).fit(X[:, ff], y)

    y_est_forward = m.predict(X[:, ff])
    residual = y - y_est_forward

    for i in range(0, len(ff)):
        title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
        figure(k+1)
        plot(X[:, ff[i]], residual, '.')
        xlabel(attributeNames[ff[i]])
        xticks([])
        ylabel('residual error')
        i += 1
        k += 1

show()