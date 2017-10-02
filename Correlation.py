from DataTransform import *

from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import corrcoef, sum, log, arange
from numpy.random import rand
from pylab import pcolor, show, colorbar, xticks, yticks


# plotting the correlation matrix. Remember transpose otherwise each row (1470)
# will be correlated

Y = X - np.ones((N, 1)) * X.std(0)
R = corrcoef(Y.T)

sns.heatmap(R[1:2].T,
            annot=True,
            annot_kws={"size":8},
            fmt='.2g',
            xticklabels=attributeNames[1:2],
            yticklabels=attributeNames)
plt.yticks(rotation=0)
plt.xticks(rotation=0)
show()
