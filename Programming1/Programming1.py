############################################################################
# Programming 1
# https://utah.instructure.com/courses/750341/assignments/10485736
#
# Authors: Marcus Corbett and Emma Kerr
#
# This script contains functions to evaluate both the Newton polynomial basis
# and Lagrange basis at a list of specified points. See main method on how
# to run this script.
#
############################################################################
import math
import numpy as np
import numpy.linalg
from matplotlib import pyplot as plt


def newton_basis(pts, degree):
    num_rows = pts.size
    mat = np.zeros((num_rows, degree + 1))
    # Fill in matrix with "unique" terms, x point difference that can not be recursively multiplied
    for i in range(0, num_rows):
        for j in range(0, degree + 1):
            if j == 0:
                mat[i, j] = 1
            elif j > i:
                mat[i, j] = 0
            else:
                mat[i, j] = (pts[i] - pts[j - 1])
    # recursively multiply each index by the column in the previous respective row
    for i in range(0, num_rows):
        for j in range(0, degree + 1):
            if j > 0:
                mat[i, j] = mat[i, j] * mat[i, j - 1]
    return mat


def lagrange_basis(pts, degree):
    num_rows = pts.size
    mat = np.zeros((num_rows, degree + 1))
    for i in range(0, num_rows):
        for j in range(0, degree + 1):
            mat[i, j] = pts[i] ** j
    return mat


def find_coefficients(x_vals, y_vals, basis):
    if basis == "newton":
        mat = newton_basis(x_vals, x_vals.shape[0] - 1)
        # Solve the linear system
        co_efs = np.linalg.solve(mat, y_vals)
        return co_efs
    if basis == "lagrange":
        mat = lagrange_basis(x_vals, x_vals.shape[0] - 1)
        # Solve the linear system
        co_efs = np.linalg.solve(mat, y_vals)
        return co_efs
    return 0


def evaluate_polynomials(co_efs, pts, basis):
    if basis == "newton":
        mat = newton_basis(pts, co_efs.shape[-1] - 1)
        # Matrix-vector multiplication
        f_vals = mat @ co_efs
        return f_vals
    if basis == "lagrange":
        mat = lagrange_basis(pts, co_efs.shape[-1] - 1)
        # Matrix-vector multiplication
        f_vals = mat @ co_efs
        return f_vals
    return 0


def evaluate_newton_basis(input_range, function):
    n = 10
    # Get equally spaced points
    start = input_range[0]
    end = input_range[1]
    xvals = np.linspace(start, end, n + 1)
    yvals = function(xvals)
    coefs = find_coefficients(xvals, yvals, "newton")

    # Evaluate the polynomial
    eval_pts = np.linspace(start, end, 500)
    pvals = evaluate_polynomials(coefs, eval_pts, "newton")

    # Plot the function and the polynomial
    plt.plot(eval_pts, pvals, linewidth=4, label='Interpolant')
    plt.plot(eval_pts, function(eval_pts), linewidth=4, linestyle='--', label='Exact')
    plt.plot(xvals, yvals, 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def evaluate_lagrange_basis(input_range, function):
    n = 10
    # Get equally spaced points
    start = input_range[0]
    end = input_range[1]
    xvals = np.linspace(start, end, n + 1)
    yvals = function(xvals)
    coefs = find_coefficients(xvals, yvals, "lagrange")

    # Evaluate the polynomial
    eval_pts = np.linspace(start, end, 500)
    pvals = evaluate_polynomials(coefs, eval_pts, "lagrange")

    # Plot the function and the polynomial
    plt.plot(eval_pts, pvals, linewidth=4, label='Interpolant')
    plt.plot(eval_pts, function(eval_pts), linewidth=4, linestyle='--', label='Exact')
    plt.plot(xvals, yvals, 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


# Function choices to interpolate
def f(x): return np.sin(x)


def h(x): return np.heaviside(x-1.5, 0.5)


def g(x): return 1.0 / (1.0 + 25.0 * (x - 1.5)**2.0)


def q(x):  # heavily oscillatory function
    return np.exp(np.cos(3 * x)) + np.sin(10 * np.sin(3 * x))


def main():
    # manipulate range plotted here
    x_range = [0, 2 * math.pi]
    # Run this to evaluate the Newton basis. Second argument is the function to interpolate.
    # You can choose functions from choices above
    evaluate_newton_basis(x_range, f)
    # Run this to evaluate the Lagrange basis. Second argument is the function to interpolate
    evaluate_lagrange_basis(x_range, f)


if __name__ == "__main__":
    main()
