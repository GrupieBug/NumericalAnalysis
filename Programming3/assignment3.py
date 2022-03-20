import math
import numpy as np
import numpy.linalg
from matplotlib import pyplot as plt


def basis(x_vals, degree, h):
    mat = np.ones((x_vals.shape[-1], degree + 1))
    for i in range(0, degree):
        # Each column is equal to the previous column times xvals.
        mat[:, i+1] = (mat[:, i] * x_vals)
    np.transpose(mat)

    for j in range(1, len(x_vals)):
        mat[:, j] = mat[:, j] / h
    return mat


def evaluate_polynomials(co_efs, pts, h):
    mat = basis(pts, co_efs.shape[-1]-1, h)
    # Matrix-vector multiplication
    f_vals = mat @ co_efs
    return f_vals


def find_weights(x_vals, y_vals, h):
    mat = basis(x_vals, x_vals.shape[0]-1, h)
    # Solve the linear system
    co_efs = np.linalg.solve(mat, y_vals)
    return co_efs


# Functions to interpolate
def f(x): return np.sin(x)


# def f(x): return np.heaviside(x-1.5, 0.5)


# def f(x): return 1.0 / (1.0 + 25.0 * (x - 1.5)**2.0)
def main():
    n = 10
    h = 1
    # Get equally spaced points
    # xvals = np.linspace(0, 3, n + 1)
    xvals = np.linspace(0, 2 * math.pi, n+1)
    for i in range(0, n):
        xvals[i] + (h * i)

    # yvals = f(xvals)
    basis_primes = np.zeros(n + 1)
    basis_primes[2] = 2/h
    weights = find_weights(xvals, basis_primes, h)

    # Evaluate the polynomial
    # ask prof
    eval_pts = np.linspace(0, 3, 10)
    pvals = evaluate_polynomials(weights, eval_pts, h)


if __name__ == "__main__":
    main()
