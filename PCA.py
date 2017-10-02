
from DataTransform import *
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd
from numpy import *

# Subtract mean value from data
X_std = StandardScaler().fit_transform(X)
#Y = (X - np.ones((N, 1)) * X.mean(0))/X_std

A = (X - mean(X, axis=0)) / std(X, axis=0)


# PCA by computing SVD of A
U, S, V = svd(A, full_matrices=False)
V = V.T

print(shape(V))
print(V)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

print('With 2 principal components we can verify that {0}% is explained'.format(sum(rho[:30 ] *100)))
# Plot variance explained
figure()
plot(range(1, len(rho) + 1), rho, 'o-')
title('Variance explained by principal components')
xlabel('Principal component')
ylabel('Variance explained')
# show()

Z = A * V

# Indices of the principal components to be plotted
i = 0
j = 1

print(A)
print(Z)

# Plot PCA of the data
f = figure()
title('IBM Attrition data: PCA')
#Z = array(Z)
#for c in range(len(classNames)):
for c in (0,1):
    # select indices belonging to class c:
    class_mask = np.array(classLabels).T == c
    plot(Z[class_mask, i], Z[class_mask, j], 'o')
legend(classNames)
xlabel('PC{0}'.format(i + 1))
ylabel('PC{0}'.format(j + 1))

# Output result to screen
show()
