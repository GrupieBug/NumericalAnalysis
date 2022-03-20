############################################################################
# Programming 2
# https://utah.instructure.com/courses/750341/assignments/10514862?confetti=true
#
# Authors: Marcus Corbett and Emma Kerr
#
# This script contains functions to evaluate the fourier basis 
# at a list of specified points. See main method on how
# to run this script.
#
############################################################################
import math
import numpy as np
import numpy.linalg
from matplotlib import pyplot as plt


def fourier_basis(eval_pts):
    """
    This function evaluates the fourier basis at the set of evaluation points
    :param eval_pts: x points to evaluate
    :return:
    """
    # Create the initial matrix
    mat_size = eval_pts.shape[-1]
    mat = np.ones((mat_size, mat_size))
    cos_eval_points = np.arange(0, mat_size // 2 + 1, dtype=np.float64)
    sin_eval_points = np.arange(1, mat_size // 2 + 1,
                                dtype=np.float64)  # these are the i input points in formula, evenly space over our
    # fixed domain. These will be multiplied by our x eval points in the cos and sin basis

    for i in range(0, mat_size):
        # This is the matrix union between the cos and sin basis terms
        mat[:, ::2] = np.cos(cos_eval_points * eval_pts[:, np.newaxis])
        mat[:, 1::2] = np.sin(sin_eval_points * eval_pts[:, np.newaxis])
    return mat


def evaluate_fourier(input_range, function):
    """
    This function is the generic method call to evaluate a set of points using the Fourier basis. Displays a final plot
    of the results
    :param input_range: the range of x values given
    :param function: the function wished to interpolate
    """
    n = 101
    # Get equally spaced points
    start = input_range[0]
    end = input_range[1]
    xvals = np.linspace(start, end, n)
    yvals = function(xvals)
    coefs = find_coefficients(xvals, yvals)

    # Evaluate the polynomial
    eval_pts = np.linspace(start, end, n)
    pvals = evaluate_polynomials(coefs, eval_pts)

    # Plot the function and the polynomial
    plt.plot(eval_pts, pvals, linewidth=4, label='Interpolant')
    plt.plot(eval_pts, function(eval_pts), linewidth=4, linestyle='--', label='Exact')
    plt.plot(xvals, yvals, 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Newton Interpolation")
    plt.show()


def find_coefficients(x_vals, y_vals):
    """
    This function was provided to solve for the coefficient matrix
    :param x_vals: input x values
    :param y_vals: values of the function evaluation points
    :return: the coefficient values of the polynomial in array form
    """
    mat = fourier_basis(x_vals)
    # Solve the linear system
    co_efs = np.linalg.solve(mat, y_vals)
    return co_efs


def evaluate_polynomials(co_efs, pts):
    """
    This function will return the final polynomial for any basis
    :param co_efs: the coefficients of the polynomial
    :param pts: the x points range to evaluate
    :return: the final polynomial equation
    """
    mat = fourier_basis(pts)
    # Matrix-vector multiplication
    f_vals = mat @ co_efs
    return f_vals


def f(x): return np.sin(x)


def main():
    # the domain we are asked to assume
    x_range = [0, 2 * math.pi]
    # Run this to evaluate the Fourier interpolation. Second argument is the function to interpolate
    # You can choose functions by manipulating f above
    evaluate_fourier(x_range, f)


if __name__ == "__main__":
    main()
