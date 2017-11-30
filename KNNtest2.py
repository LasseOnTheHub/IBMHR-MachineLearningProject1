from DataTransform import *

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from numpy import *

#Standardize the data
X = (X - mean(X, axis=0)) / std(X, axis=0)

y = np.squeeze(np.asarray(y))
loops = 1

max_k = np.arange(1, 25)
scores = np.zeros(len(max_k))

#
# Computes the optimal value of k by using two-layer cross-validation.
#
for k in max_k:
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=k, p=1)

    scores[k - 1] = 1 - cross_val_score(clf, X=X, y=y, cv=cv).mean()

print('Optimal value of k: {0}'.format(np.where(scores == min(scores))[0][0]))

plt.figure()
plt.plot(max_k, scores, 'r-')
plt.xlabel('Number of neighbors')
plt.ylabel('Classification error rate (%)')
plt.show()
