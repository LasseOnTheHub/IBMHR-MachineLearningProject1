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

    res.to_excel('output.xlsx')

    return data

def getMatrixFromXlsx(path):
    # Load xls sheet with data
    doc = xlrd.open_workbook(path).sheet_by_index(0)

    # Extract attribute names (1st row, column 4 to 12)
    attributeNames = doc.row_values(0, 1, 49)

    # Extract class names to python list,
    # then encode with integers (dict)
    classLabels = doc.col_values(2, 2, 1470)
    classNames = sorted(set(classLabels))
    classDict = dict(zip(classNames, range(5)))

    # Extract vector y, convert to NumPy matrix and transpose
    y = np.mat([classDict[value] for value in classLabels]).T

    # Preallocate memory, then extract excel data to matrix X
    X = np.mat(np.empty((1470, 50)))
    for i, col_id in enumerate(range(2, 51)):
        X[:, i] = np.mat(doc.col_values(col_id, 1, 1471)).T

    # Compute values of N, M and C.
    N = len(y)
    M = len(attributeNames)
    C = len(classNames)

    print(classLabels)
    print(classDict)
    print(X);
    print(X.shape)
    print(attributeNames)

# Not used ATM.
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
    classLabels = data.loc[:, 'Attrition':'Attrition']
    classNames = sorted(set(classLabels))
    classDict = dict(zip(classNames, range(2)))




    # Extract vector y, convert to NumPy matrix and transpose
    y = np.array([classDict[value] for value in classLabels])

    # Preallocate memory, then extract excel data to matrix X
    #X = np.mat(np.empty((1470, 50)))
    #for i, col_id in enumerate(range(1, 49)):
    #    X[:, i] = np.mat(data.loc[:, 'Age':'MaritalStatus_Single']).T

    # Compute values of N, M and C.
    N = len(y)
    M = len(attributNames)
    C = len(classNames)

    print(classLabels)