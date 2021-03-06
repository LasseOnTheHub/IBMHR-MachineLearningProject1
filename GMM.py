# exercise 11.1.5
from matplotlib.pyplot import figure, plot, legend, xlabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from DataTransform import *
from numpy import *

#Extract interesting continous variables.
X = X[:, [2, 3, 6, 10, 13, 49]]
#X = X[:, [1, 49]]
#Standardize the data
X = (X - mean(X, axis=0)) / std(X, axis=0)
y = np.squeeze(np.asarray(y).T)
print(X.shape)

# Range of K's to try
KRange = range(1, 13)
T = len(KRange)

covar_type = 'full'  # you can try out 'diag' as well
reps = 10  # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10, shuffle=True)

for t, K in enumerate(KRange):
    print('Fitting model for K={0}'.format(K))

    # Fit Gaussian mixture model
    gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)

    # Get BIC and AIC
    BIC[t,] = gmm.bic(X)
    AIC[t,] = gmm.aic(X)

    # For each crossvalidation fold
    for train_index, test_index in CV.split(X):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        X_test = X[test_index]

        # Fit Gaussian mixture model to X_train
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

        # compute negative log likelihood of X_test
        CVE[t] += -gmm.score_samples(X_test).sum()

# Plot results

NLOGL = -gmm.score(X_test).sum()

print(NLOGL)

figure(1);
plot(KRange, BIC, '-*b')
plot(KRange, AIC, '-xr')
plot(KRange, 2 * CVE, '-ok')
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
show()

print('Ran Exercise 11.1.5')