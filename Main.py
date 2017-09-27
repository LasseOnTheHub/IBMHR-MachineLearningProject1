# Main file for Machine Learning Project 1
# Run needed functions from this file.

from Functions import preprocessing, computePCA, getMatrixFromXlsx


preprocessing('Data/IBMData.csv')
data = getMatrixFromXlsx('output.xlsx')
computePCA(data)

