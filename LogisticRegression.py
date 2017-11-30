from DataTransform import *
import itertools
from numpy import *

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold, train_test_split

import matplotlib.pyplot as plt

# Load Matlab data file and extract variables of interest
X = X[:,1:51]
y = np.squeeze(np.asarray(y))
attributeNames = attributeNames[1:51]
N, M = X.shape
C = len(classNames)

#Standardize the data
X = (X - mean(X, axis=0)) / std(X, axis=0)

folds = 15
cv = KFold(folds)

nested_scores = np.zeros(folds)
non_nested_score = np.zeros(folds)

X_outer_train, X_outer_test, y_outer_train, y_outer_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

for i, (train_indices, validation_indices) in enumerate(cv.split(X_outer_train, y_outer_train)):

    X_inner_train, X_inner_test = X[train_indices], X[validation_indices]
    y_inner_train, y_inner_test = y[train_indices], y[validation_indices]

    model = LogisticRegression()
    model.fit(X_inner_train, y_inner_train)

    inner_test_predicted = model.predict(X_inner_test)
    outer_test_predicted = model.predict(X_outer_test)

    inner_score_accuracy = accuracy_score(y_inner_test, inner_test_predicted)
    outer_score_accuracy = accuracy_score(y_outer_test, outer_test_predicted)

    non_nested_score[i] = (1 - accuracy_score(y_inner_test, inner_test_predicted)) * 100
    nested_scores[i] = (1 - accuracy_score(y_outer_test, outer_test_predicted)) * 100

print('Average non-nested score: {0:.4f}'.format(non_nested_score.mean()))
print('Average nested score: {0:.4f}'.format(nested_scores.mean()))

plt.figure()

plt.plot(range(folds), nested_scores, '.b-')
plt.plot(range(folds), non_nested_score, '.r-')

plt.xlabel('Number of K-Folds')
plt.ylabel('Classification error rate (%)')

plt.title('Classification error rate of logistic regression')

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

clf = LogisticRegression()
clf = clf.fit(X_train, y_train)

cm = confusion_matrix(y_test, clf.predict(X_test))

print(cm)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual results')
    plt.xlabel('Predicted results')


classes = ['No attrition', 'Attrition']

plt.figure()
plot_confusion_matrix(cm, classes, False, 'Logistic regression of attrition')
plt.show()