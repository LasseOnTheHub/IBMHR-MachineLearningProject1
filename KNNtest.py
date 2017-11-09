# Load required packages
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from DataTransform import *


# Create a scaler object
sc = StandardScaler()

# Fit the scaler to the feature data and transform
X_std = sc.fit_transform(X)

# Create a list of 10 candidate values for the C parameter
C_candidates = dict(C=np.logspace(-4, 4, 10))


knclassifier = KNeighborsClassifier(n_neighbors=l,p=1);
knclassifier.fit(X, y);

# Create a gridsearch object with the support vector classifier and the C value candidates
#clf = GridSearchCV(estimator=SVC(), param_grid=C_candidates)

# Fit the cross validated grid search on the data
#clf.fit(X_std, y)

# Show the best value for C
print(knclassifier.effective_metric_params_)

print(cross_val_score(knclassifier, X_std, y))