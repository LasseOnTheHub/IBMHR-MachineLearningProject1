from RegressionCombined import *
from DataTransform2 import *
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot, title
import numpy as np
from scipy import stats

actual = np.asarray(y_test)
ann = np.asarray(y_est)
forward = np.asarray(y_est_for)
avg_guess = np.empty(len(y_test))
avg_guess.fill(y.mean(0))

actual_line = actual - actual
ann_line = ann.reshape(-1,1) - actual
forward_line = forward.reshape(-1,1) - actual
avg_guess_line = avg_guess.reshape(-1, 1) - actual
avg_error = Error_avg.reshape(-1, 1)
Error_test_fs_for = Error_test_fs.reshape(-1, 1)
errors_ann = errors_ann.reshape(-1, 1)

f = figure(1);
f.hold(True)
title('Predicted')
plot(actual)
plot(ann)
plot(forward)
plot(avg_guess)
xlabel('Observation')
ylabel('Predicted')
legend(['actual', 'ann', 'forward', 'avg_guess'])


f = figure(2);
f.hold(True)
title('Squarre error')
plot(actual_line)
plot(ann_line)
plot(forward_line)
plot(avg_guess_line)
xlabel('Observation')
ylabel('Error')
legend(['actual', 'ann', 'forward', 'avg_guess'])


mean_squarre_ann = (sum(np.square(ann_line))) / y_test.shape[0]
mean_squarre_forward = (sum(np.square(forward_line))) / y_test.shape[0]
mean_squarre_avg = (sum(np.square(avg_guess_line))) / y_test.shape[0]

z1 = (errors_ann - Error_test_fs_for)
zb1 = z1.mean()
nu1 = K - 1
sig1 = (z1 - zb1).std() / np.sqrt(K - 1)
alpha1 = 0.05

zL1 = zb1 + sig1 * stats.t.ppf(alpha1 / 2, nu1);
zH1 = zb1 + sig1 * stats.t.ppf(1 - alpha1 / 2, nu1);
print('ANN vs linear regression')
if zL1 <= 0 and zH1 >= 0:
    print('Models are not significantly different')
else:
    print('Model are significantly different.')

# Boxplot to compare classifier error distributions
figure(3)
title('Box-plot over error')
boxplot(np.bmat('errors_ann, Error_test_fs_for, avg_error'))
xlabel('Neural network vs. linear regression vs. average guessing')
ylabel('Cross-validation error')


z2 = (errors_ann - avg_error)
zb2 = z2.mean()
nu2 = K - 1
sig2 = (z2 - zb2).std() / np.sqrt(K - 1)
alpha2 = 0.05

zL2 = zb2 + sig2 * stats.t.ppf(alpha2 / 2, nu2);
zH2 = zb2 + sig2 * stats.t.ppf(1 - alpha2 / 2, nu2);
print('ANN vs average guessing')
if zL2 <= 0 and zH2 >= 0:
    print('Models are not significantly different')
else:
    print('Model are significantly different.')

z3 = (Error_test_fs_for - avg_error)
zb3 = z3.mean()
nu3 = K - 1
sig3 = (z3 - zb3).std() / np.sqrt(K - 1)
alpha3 = 0.05

zL3 = zb3 + sig3 * stats.t.ppf(alpha3 / 2, nu3);
zH3 = zb3 + sig3 * stats.t.ppf(1 - alpha3 / 2, nu3);
print('Linear regression vs average guessing')
if zL3 <= 0 and zH3 >= 0:
    print('Models are not significantly different')
else:
    print('Model are significantly different.')


show()
