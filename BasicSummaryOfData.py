from DataTransform import *
import pandas as pd

A = np.squeeze(np.asarray(X))

# Convert to panda dataframe
P = pd.DataFrame(data=X,
             index=indexNumber,
             columns=attributeNames)


xlsxpath = 'summarystatistics.xlsx'
PSummary = P.describe().T
PSummary.to_excel(xlsxpath)