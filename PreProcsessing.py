import pandas as pd

# Load csv data file
data = pd.DataFrame.from_csv('Data/IBMData.csv',
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
xlsxpath = 'output.xlsx'
res.to_excel(xlsxpath)