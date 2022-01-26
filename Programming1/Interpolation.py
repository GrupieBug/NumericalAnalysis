import numpy as np
import numpy.linalg
from matplotlib import pyplot as plt


def monomial_basis(x_vals, degree):
    mat = np.ones((x_vals.shape[-1], degree + 1))
    for i in range(0, degree):
        # Each column is equal to the previous column times xvals.
        mat[:, i+1] = mat[:, i] * x_vals
    return mat


def evaluate_polynomials(co_efs, pts):
    mat = monomial_basis(pts, co_efs.shape[-1]-1)
    # Matrix-vector multiplication
    f_vals = mat @ co_efs
    return f_vals


def find_coefficients(x_vals, y_vals):
    mat = monomial_basis(x_vals, x_vals.shape[0]-1)
    # Solve the linear system
    co_efs = np.linalg.solve(mat, y_vals)
    return co_efs


# Functions to interpolate
def f(x): return np.sin(x)


# def f(x): return np.heaviside(x-1.5, 0.5)


# def f(x): return 1.0 / (1.0 + 25.0 * (x - 1.5)**2.0)
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
