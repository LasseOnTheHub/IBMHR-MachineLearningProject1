from PreProcsessing import *

import xlrd
import numpy as np

#doc = xlrd.open_workbook('output2.xlsx').sheet_by_index(0)

# Extract attribute names (1st row, column 3 to 51)
#attributeNames = doc.row_values(0, 0, 51)
#indexNumber = doc.col_values(0,1)

# Preallocate memory, then extract excel data to matrix X
#X = np.mat(np.empty((1470, 51), dtype=int))
#for i, col_id in enumerate(range(0, 50)):
#    X[:, i] = np.mat(doc.col_values(col_id, 1, 1471)).T

#y = X[:,50]

#del X["MonthlyIncome"]
#np.savetxt('testX2.txt', X, delimiter=',')
# Compute values of N, M and C.
#N = len(y)
#M = len(attributeNames)
#C = len(classNames)

from scipy.stats import zscore
import pandas as pd

# Load our datafile
data = pd.read_csv("Data/IBMData.csv", index_col=9)

# Remove unused columns

del data["EmployeeCount"]
del data["Over18"]

# Convert yes/no to binary
data.replace(('Yes', 'No'), (1, 0), inplace=True)

# Do a one-out-of-k on all categorical columns #LarsErEnBredDame
data = pd.get_dummies(data)

# Save transformed data in a new file
data.to_csv('Data/transformed.csv')


y = data.MonthlyIncome.values
X = data.ix[:, data.columns != "MonthlyIncome"]

attributeNames = X.columns
X = X.values

X = X / np.linalg.norm(X)

N, M = X.shape
