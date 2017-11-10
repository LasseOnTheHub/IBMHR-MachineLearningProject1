from DataTransform import *

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

y = np.squeeze(np.asarray(y))

loops = 5

k = np.arange(5, 20)

estimator = KNeighborsClassifier(p=1)
parameter = {'n_neighbors': k}

non_nested_scores = np.zeros(loops)
nested_scores = np.zeros(loops)
best_parameters = np.zeros(loops)

for i in range(loops):

    print('Running loop {0}/{1}'.format(i + 1, loops))

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

    clf = GridSearchCV(estimator=estimator, param_grid=parameter, cv=inner_cv)
    clf.fit(X, y)

    best_parameters[i] = clf.best_params_['n_neighbors']
    non_nested_scores[i] = clf.best_score_

    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    nested_scores[i] = nested_score.mean()
    print(clf.best_params_)


score_difference = non_nested_scores - nested_scores
print('Average score difference:{0}, std: {1}'.format(score_difference.mean(), score_difference.std()))
print('Nested score: {0}'.format(nested_scores.mean()))
print('Non-Nested score: {0}'.format(non_nested_scores.mean()))
