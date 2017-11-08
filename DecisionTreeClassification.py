from DataTransform import *
from PreProcsessing import *

import numpy as np
from sklearn import tree

#Remove the attrition attribute from the data X
X = X[:,1:51]
attributeNames = attributeNames[1:51]


print(len(attributeNames))
print(X.shape)

# Update N and M
N, M = X.shape

# Fit regression tree classifier, Gini split criterion, no pruning
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributeNames)

