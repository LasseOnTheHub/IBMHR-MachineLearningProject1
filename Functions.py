# Functions

import xlrd
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show
from scipy.linalg import svd

N = 0


def preprocessing ( path ):
    # Load csv data file
    data = pd.DataFrame.from_csv(path,
                                 header=0,
                                 sep=',',
                                 index_col=9,
                                 parse_dates=True,
                                 encoding=None,
                                 tupleize_cols=False,
                                 infer_datetime_format=False)

    # Delete unused columns
    del data['Over18']
    del data['StockOptionLevel']
    del data['EmployeeCount']
    del data['StandardHours']

    # Make binary yes/no value into 1/0 values (Prevent attrition to be split up)
    data.replace(('Yes', 'No'), (1, 0), inplace=True)

    # transform categorical data to one-out-of-k
    res = pd.get_dummies(data)
    # output transformed file.

    res.to_excel('output.xlsx')



def getMatrixFromXlsx(path):
    # Load xls sheet with data
    doc = xlrd.open_workbook(path).sheet_by_index(0)

    # Extract attribute names (1st row, column 4 to 12)
    attributeNames = doc.row_values(0, 1, 50)

    # Extract class names to python list,
    # then encode with integers (dict)
    classLabels = doc.col_values(2, 1, 1470)
    classNames = sorted(set(classLabels))
    classDict = dict(zip(classNames, range(5)))

    # Extract vector y, convert to NumPy matrix and transpose
    y = np.mat([classDict[value] for value in classLabels]).T

    # Preallocate memory, then extract excel data to matrix X
    X = np.mat(np.empty((1469, 50),dtype=int))
    for i, col_id in enumerate(range(1, 51)):
        X[:, i] = np.mat(doc.col_values(col_id, 1, 1470)).T

    np.savetxt('testX2.txt', X, delimiter=',')
    # Compute values of N, M and C.
    global N
    N = len(y)
    M = len(attributeNames)
    C = len(classNames)

    print(X.dtype)
    print(y.shape)
    print(X.shape)
    return X

# Compute PCA
def computePCA(X):
    # Subtract mean value from data
    global N
    Y = X - np.ones((N, 1)) * X.mean(0)

    np.savetxt('testX.txt', X, delimiter=',')
    np.savetxt('testY.txt', Y,delimiter=',')
    print('her kommer Y')
    print(Y.shape)
    print(X.shape)
    print(np.argwhere(np.isnan(Y)))

    # PCA by computing SVD of Y
    U, S, V = svd(Y, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()

    # Plot variance explained
    figure()
    plot(range(1, len(rho) + 1), rho, 'o-')
    title('Variance explained by principal components');
    xlabel('Principal component');
    ylabel('Variance explained');
    show()