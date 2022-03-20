import numpy as np
import numpy.linalg
from matplotlib import pyplot as plt

# Functions and data points to approximate
# f = lambda x : np.exp(-x*x)
# fp = lambda x : -2 * x * np.exp(-x*x)
f = lambda x: np.sin(x)
fp = lambda x: np.cos(x)
xhat = 1.0
# Number and array of spacing between points.
N = 50
hh = np.logspace(-3, 0, N)
# Arrays to hold our errors
err1 = 0 * hh
err2 = 0 * hh
err3 = 0 * hh
# Loop through our 'h' values
for i in range(0, len(hh)):
    h = hh[i]

    # Three differen finite difference approximations
    fd1 = 1 / h * (f(xhat + h) - f(xhat))
    fd2 = -1 / h * (f(xhat + h) - 4.0 * f(xhat + h * 0.5) + 3.0 * f(xhat))
    fd3 = -1 / (3.0 * h) * (21.0 * f(xhat) - 32.0 * f(xhat + 0.25 * h) + 12.0 * f(xhat
                                                                                  + 0.5 * h) - f(xhat + h))

    # Compute errors
    err1[i] = abs(fd1 - fp(xhat))
    err2[i] = abs(fd2 - fp(xhat))
    err3[i] = abs(fd3 - fp(xhat))
# Estimate coefficients of the line of best fit
p1 = np.polyfit(np.log(hh), np.log(err1), 1)
p2 = np.polyfit(np.log(hh), np.log(err2), 1)
p3 = np.polyfit(np.log(hh), np.log(err3), 1)
# Make a pretty plot
plt.plot(hh, err1, linewidth=4, label='Order = ' + str(p1[0]))
plt.plot(hh, err2, linewidth=4, label='Order = ' + str(p2[0]))
plt.plot(hh, err3, linewidth=4, label='Order = ' + str(p3[0]))
plt.xlabel('h')
plt.ylabel('Absolute error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()