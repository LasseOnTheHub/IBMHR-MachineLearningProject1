# Main file for Machine Learning Project 1
# Run needed functions from this file.

from Functions import preprocessing, getMatrixFromCSV, getMatrixFromXlsx


data = preprocessing('Data\IBMData.csv')
getMatrixFromXlsx('output.xlsx')
