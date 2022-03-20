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
    """
    This function returns the basis for the Newton polynomial interpolation

    I'm keeping this here because I'm not sure about the accuracy of the "newton_2" function. This can be used instead
    to find coef
    instead in the find coef function because it is accurate.
    :param pts: the set of x-points to interpolate from
    :param degree: the degree of the polynomial of which to interpolate
    :return: the basis matrix for the Newton polynomial
    """
    num_rows = pts.size
    mat = np.zeros((num_rows, degree + 1))

    # Fill in matrix with "unique" terms, x point difference that can not be "recursively" multiplied
    # This results in a lower triangular system
    for i in range(0, num_rows):
        for j in range(0, degree + 1):
            if j == 0:
                # upper triangle of the matrix should be zero
                mat[i, j] = 1
            elif j > i:
                # first column must be 1
                mat[i, j] = 0
            else:
                # otherwise, the "unique" term is as follows:
                mat[i, j] = (pts[i] - pts[j - 1])
    # "recursively" multiply each index by the column in the previous respective row
    for i in range(0, num_rows):
        for j in range(0, degree + 1):
            if j > 0:
                # the values at each column are the results from the previous column multiplied by itself
                mat[i, j] = mat[i, j] * mat[i, j - 1]
    return mat


def newton_2(pts, interp):
    """
    This is the second implementation of the newton method. This is a re-correction
    :param pts: The x-points to evaluate
    :param interp: The interpolation points to evaluate (set = to x-points to find coefficients
    :return: A matrix of the basis function
    """
    mat = np.ones((pts.shape[-1], interp.shape[-1]))
    for i in range(0, interp.shape[-1] - 1):
        # Each column is equal to the previous column times difference in x-vals to interpolation points
        mat[:, i + 1] = mat[:, i] * (pts - interp[i])
    return mat


def lagrange_basis(pts, degree):
    """
    This function returns the matrix that contains the basis for the lagrange polynomial
    :param pts: interpolation x inputs
    :param degree: degree of the resulting polynomial
    :return: Legrange matrix that can be used to solve for the legrange polynomials
    """
    num_rows = pts.size
    mat = np.ones((num_rows, degree + 1))
    row_count = 0
    for i in range(0, degree - 1):
        for j in range(0, num_rows):
            if j is not i:
                # Each column is equal to the previous column times difference in x-vals to interpolation points
                mat[j, i + 1] = mat[j, i] * ((0 - pts[i + 1]) / (pts[i + 1] - pts[j]))
    return mat


def fourier_basis(eval_pts, interp_pts):
    # Create the initial matrix
    mat_size = eval_pts.shape[-1]
    mat = np.ones((mat_size, mat_size))
    cos_eval_points = np.arange(0, mat_size // 2 + 1, dtype=np.float64)
    sin_eval_points = np.arange(1, mat_size // 2 + 1,
                                dtype=np.float64)  # these are the i input points in formula, evenly space over our
    # fixed domain

    for i in range(0, mat_size):
        # The next basis fcn is the previous times a new linear polynomial
        mat[:, ::2] = np.cos(cos_eval_points * eval_pts[:, np.newaxis])
        mat[:, 1::2] = np.sin(sin_eval_points * eval_pts[:, np.newaxis])
    return mat


def find_coefficients(x_vals, y_vals, basis):
    """
    This function was provided to solve for the coefficient matrix
    :param x_vals: input x values
    :param y_vals: values of the function evaluation points
    :param basis: given basis matrix
    :return: the coefficient values of the polynomial in array form
    """
    if basis == "newton":
        mat2 = newton_basis(x_vals, x_vals.shape[0] - 1)
        mat = newton_2(x_vals, x_vals)
        # Solve the linear system
        co_efs = np.linalg.solve(mat, y_vals)
        return co_efs
    if basis == "lagrange":
        mat = lagrange_basis(x_vals, x_vals.shape[0] - 1)
        # Solve the linear system
        co_efs = np.linalg.solve(mat, y_vals)
        return co_efs
    return 0


def evaluate_polynomials(co_efs, pts, interp_points, basis):
    """
    This function will return the final polynomial for any basis
    :param interp_points:
    :param co_efs: the coefficients of the polynomial
    :param pts: the x input points
    :param basis: the basis function solved
    :return: the final polynomial equation
    """
    if basis == "newton":
        mat = newton_2(pts, interp_points)
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
    """
    This function is the generic method call to evaluate a set of points using the newton basis. Displays a final plot
    of the results
    :param input_range: the range of x values given
    :param function: the function wished to interpolate
    """
    n = 10
    # Get equally spaced points
    start = input_range[0]
    end = input_range[1]
    xvals = np.linspace(start, end, n + 1)
    yvals = function(xvals)
    coefs = find_coefficients(xvals, yvals, "newton")

    # Evaluate the polynomial
    eval_pts = np.linspace(start, end, 500)
    x_new = np.linspace(start, end, n + 1)
    pvals = evaluate_polynomials(coefs, eval_pts, x_new, "newton")

    # Plot the function and the polynomial
    plt.plot(eval_pts, pvals, linewidth=4, label='Interpolant')
    plt.plot(eval_pts, function(eval_pts), linewidth=4, linestyle='--', label='Exact')
    plt.plot(xvals, yvals, 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Newton Interpolation")
    plt.show()


def evaluate_lagrange_basis(input_range, function):
    """
    This function is the generic method call to evaluate a set of points using the Lagrange basis. Displays a final plot
    of the results
    :param input_range: the range of x values given
    :param function: the function wished to interpolate
    """
    n = 10
    # Get equally spaced points
    start = input_range[0]
    end = input_range[1]
    xvals = np.linspace(start, end, n + 1)
    yvals = function(xvals)
    coefs = find_coefficients(xvals, yvals, "lagrange")

    # Evaluate the polynomial
    eval_pts = np.linspace(start, end, 500)
    pvals = evaluate_polynomials(coefs, eval_pts, [], "lagrange")

    # Plot the function and the polynomial
    plt.plot(eval_pts, pvals, linewidth=4, label='Interpolant')
    plt.plot(eval_pts, function(eval_pts), linewidth=4, linestyle='--', label='Exact')
    plt.plot(xvals, yvals, 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Lagrange Interpolation")
    plt.show()


# Function choices to interpolate
def f(x): return np.sin(x)


def h(x): return np.heaviside(x - 1.5, 0.5)


def g(x): return 1.0 / (1.0 + 25.0 * (x - 1.5) ** 2.0)


def q(x):  # heavily oscillatory function
    return np.exp(np.cos(3 * x)) + np.sin(10 * np.sin(3 * x))


def main():
    # manipulate range plotted here
    x_range = [0, 2 * math.pi]
    # Run this to evaluate the Newton basis. Second argument is the function to interpolate passed as a delegate
    # You can choose functions from choices above
    evaluate_newton_basis(x_range, f)
    # Run this to evaluate the Lagrange basis. Second argument is the function to interpolate passed as a delegate
    evaluate_lagrange_basis(x_range, f)


if __name__ == "__main__":
    main()
