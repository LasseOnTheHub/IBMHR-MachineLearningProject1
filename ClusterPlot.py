# exercise 11.1.1
from matplotlib.pyplot import figure, show
import numpy as np
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.mixture import GaussianMixture
from DataTransform import *
from numpy import *

# Load Matlab data file and extract variables of interest
#Extract interesting continous variables.
X = X[:, [2, 3, 6, 10, 13, 49]]

#X = X[:, [1, 49]]

print(X)
#Standardize the data
X = (X - mean(X, axis=0)) / std(X, axis=0)
y = np.squeeze(np.asarray(y))

N, M = X.shape



C = len(classNames)
# Number of clusters
K = 8
cov_type = 'diag'
# type of covariance, you can try out 'diag' as well
reps = 1
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(X)
cls = gmm.predict(X)
# extract cluster labels
cds = gmm.means_
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type == 'diag':
    new_covs = np.zeros([K, M, M])

count = 0
for elem in covs:
    temp_m = np.zeros([M, M])
    for i in range(len(elem)):
        temp_m[i][i] = elem[i]

    new_covs[count] = temp_m
    count += 1

covs = new_covs
# Plot results:
#figure(figsize=(14, 9))
#clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
#show()

# In case the number of features != 2, then a subset of features most be plotted instead.
figure(figsize=(14,9))
idx = [1,5] # feature index, choose two features to use as x and y axis in the plot
clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
show()

print('Ran Exercise 11.1.1')