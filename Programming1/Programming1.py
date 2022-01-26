############################################################################
# Programming 1
# https://utah.instructure.com/courses/750341/assignments/10485736
#
# Authors: Marcus Corbett and Emma Kerr
#
# This script contains functions to evaluate both the Newton polynomial basis
# and Lagrange basis at a list of specified points
#
############################################################################
import math
import numpy as np
import numpy.linalg
from matplotlib import pyplot as plt


def newton_basis(pts, co_efs):
    return 0


def lagrange_basis():
    return 0


def evaluate_polynomials(co_efs, pts):
    mat = newton_basis(pts, co_efs.shape[-1]-1)
    # Matrix-vector multiplication
    f_vals = mat @ co_efs
    return f_vals


def find_coefficients(x_vals, y_vals):
    mat = newton_basis(x_vals, x_vals.shape[0]-1)
    # Solve the linear system
    co_efs = np.linalg.solve(mat, y_vals)
    return co_efs


# Functions to interpolate
def f(x): return np.sin(x)


# def f(x): return np.heaviside(x-1.5, 0.5)


# def f(x): return 1.0 / (1.0 + 25.0 * (x - 1.5)**2.0)


def heavily_oscillatory_function(x):
    return math.exp(math.cos(3 * x)) + math.sin(10 * math.sin(3 * x))


def main():
    n = 10
    # Get equally spaced points
    xvals = np.linspace(0, 3, n + 1)
    yvals = f(xvals)
    coefs = find_coefficients(xvals, yvals)

    # Evaluate the polynomial
    eval_pts = np.linspace(0, 3, 500)
    pvals = evaluate_polynomials(coefs, eval_pts)

    # Plot the function and the polynomial
    plt.plot(eval_pts, pvals, linewidth=4, label='Interpolant')
    plt.plot(eval_pts, f(eval_pts), linewidth=4, linestyle='--', label='Exact')
    plt.plot(xvals, yvals, 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

