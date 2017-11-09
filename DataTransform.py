from PreProcsessing import *

import xlrd
import numpy as np

doc = xlrd.open_workbook('output2.xlsx').sheet_by_index(0)

# Extract attribute names (1st row, column 3 to 51)
attributeNames = doc.row_values(0, 1, 51)
indexNumber = doc.col_values(0,1)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(1, 1, 1471)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

# Extract vector y, convert to NumPy matrix and transpose
y = np.mat([classDict[value] for value in classLabels]).T

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((1470, 50), dtype=int))
for i, col_id in enumerate(range(1, 51)):
    X[:, i] = np.mat(doc.col_values(col_id, 1, 1471)).T

#np.savetxt('testX2.txt', X, delimiter=',')
# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

