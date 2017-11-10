from DataTransform import *
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn import tree

X = X[:,1:51]
y = np.squeeze(np.asarray(y))
attributeNames = attributeNames[1:51]

loops = 5
dtc = tree.DecisionTreeClassifier(criterion='gini')
params = {'max_depth':range(1,6), 'min_samples_split':range(40,130)}
#params = {'max_depth':range(1,6)}

non_nested_scores = np.zeros(loops)
nested_scores = np.zeros(loops)
best_params = np.zeros(loops)

for i in range(loops):
    print('Running loop {0}'.format(i))
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

    clf = GridSearchCV(estimator=dtc,param_grid=params,cv=inner_cv)
    clf.fit(X,y)

    non_nested_scores[i] = clf.best_score_

    print(clf.best_params_)

    nested_scores[i] = cross_val_score(clf, X,y,cv=outer_cv).mean()

score_difference = non_nested_scores - nested_scores
print('Average score difference:{0}, std: {1}'.format(score_difference.mean(), score_difference.std()))
print('Nested score: {0}'.format(nested_scores.mean()))
print('Non-Nested score: {0}'.format(non_nested_scores.mean()))

