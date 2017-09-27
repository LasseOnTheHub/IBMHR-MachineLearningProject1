# Functions

import xlrd
import numpy as np
import pandas as pd


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

    res.to_csv('output.csv')

    return data

def getMatrixFromCSV(path):

    data = pd.DataFrame.from_csv(path,
                                 header=0,
                                 sep=',',
                                 index_col=0,
                                 parse_dates=True,
                                 encoding=None,
                                 tupleize_cols=False,
                                 infer_datetime_format=False)

    attributNames = list(data.columns.values)
    classLabels = data.loc[:, 'Attrition']
    classNames = sorted(set(classLabels))
    classDict = dict(zip(classNames, range(2)))

    print(data)

    print(attributNames)
    print(classDict)

    # Extract vector y, convert to NumPy matrix and transpose
    y = np.array([classDict[value] for value in classLabels])

    # Preallocate memory, then extract excel data to matrix X
    X = np.mat(np.empty((1470, 50)))
    for i, col_id in enumerate(range(1, 49)):
        X[:, i] = np.mat(data.loc[:, 'Age':'MaritalStatus_Single']).T

    # Compute values of N, M and C.
    N = len(y)
    M = len(attributNames)
    C = len(classNames)

    print(classNames)
    print(classLabels)