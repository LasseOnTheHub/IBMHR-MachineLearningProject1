from DataTransform import *

import numpy as np
from sklearn import tree

#Remove the attrition attribute from the data X
X = X[:,1:51]
attributeNames = attributeNames[1:51]
print(attributeNames)

print(len(attributeNames))
print(X.shape[0])
print(X.T.shape)
print(y)

# Update N and M
N, M = X.shape

# Fit regression tree classifier, Gini split criterion, no pruning
dtc = tree.DecisionTreeClassifier(criterion='gini',max_depth=4, min_samples_split=63)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributeNames)

