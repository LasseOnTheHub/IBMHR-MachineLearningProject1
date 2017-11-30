import sklearn.linear_model as lm
from sklearn import cross_validation
from DataTransform import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

classes = ['No attrition', 'Attrition']

X = X[:,1:51]
y = np.squeeze(np.asarray(y))
attributeNames = attributeNames[1:51]
N, M = X.shape
C = len(classNames)

K = 5
CV = cross_validation.KFold(N, K, shuffle=True)
# Initialize variables
Error_logreg = np.empty((K, 1))
Predicted = np.empty((K,1))
Tested = np.empty((K,1))
n_tested = 0
k = 0

for train_index, test_index in CV:
    print('CV-fold {0} of {1}'.format(k + 1, K))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    # Fit and evaluate Logistic Regression classifier
    model = lm.logistic.LogisticRegression(C=N)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    Error_logreg[k] = 100 * (y_pred != y_test).sum().astype(float) / len(y_test)

    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test,y_pred), classes, False, 'Logistic regression')
    k += 1

print(Error_logreg.mean())
plt.show()