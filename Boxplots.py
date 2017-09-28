from DataTransform import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show)
from scipy.stats import zscore
import numpy as np
from scipy.io import loadmat


# Convert to panda dataframe and standardize data
P = pd.DataFrame(data=zscore(X,ddof=1),
             index=indexNumber,
             columns=attributeNames)

#figure(figsize=(12,6))
#title('Wine: Boxplot (standarized)')
#boxplot(zscore(P, ddof=1), attributeNames)
#xticks(range(1,M+1), attributeNames, rotation=90)



plt.figure();

# Sorry for the huge amount of arguments.
bp = P.boxplot(column=['Age',
'DailyRate',
'DistanceFromHome',
'Education',
'EnvironmentSatisfaction',
'HourlyRate',
'JobInvolvement',
'JobLevel',
'JobSatisfaction',
'MonthlyIncome',
'MonthlyRate',
'NumCompaniesWorked',
'OverTime',
'PercentSalaryHike',
'PerformanceRating',
'RelationshipSatisfaction',
'TotalWorkingYears',
'TrainingTimesLastYear',
'WorkLifeBalance',
'YearsAtCompany',
'YearsInCurrentRole',
'YearsSinceLastPromotion',
'YearsWithCurrManager',])

plt.xticks(rotation=90)
plt.title('IBM HR Boxplot (standardized)')

# There could be outliers in : NumCompaniesWorked TrainingTimesLastYear YearsInCurrentRole
# YearsSinceLastPromotion YearsWithCurrManager

figure(figsize=(20,15))
u = np.floor(np.sqrt(M));
v = np.ceil(float(M)/u)
for i in range(M-26):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('IBM HR: Histogram (selected)')

# not really any outlier, but will plot some specific just for the example of it

figure(figsize=(14,9))
m = [12, 18, 21,22,23]
for i in range(len(m)):
    subplot(1,len(m),i+1)
    hist(X[:,m[i]],50)
    xlabel(attributeNames[m[i]])
    ylim(0, 650) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i == 0: title('IBM HR: Histogram (selected)')
show()