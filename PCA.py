
from DataTransform import *

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd

# Subtract mean value from data
Y = X - np.ones((N, 1)) * X.mean(0)

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

print('With 2 principal components we can verify that {0}% is explained'.format(sum(rho[:2 ] *100)))
# Plot variance explained
figure()
plot(range(1, len(rho) + 1), rho, 'o-')
title('Variance explained by principal components')
xlabel('Principal component')
ylabel('Variance explained')
# show()

Z = Y * V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
f.hold()
title('IBM Attrition data: PCA')
# Z = array(Z)
for c in range(len(classNames)):
    # select indices belonging to class c:
    class_mask = y.A.ravel() == c
    plot(Z[class_mask, i], Z[class_mask, j], 'o')
legend(classNames)
xlabel('PC{0}'.format(i + 1))
ylabel('PC{0}'.format(j + 1))

# Output result to screen
show()
