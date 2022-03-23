import math
import numpy as np
import numpy.linalg
from matplotlib import pyplot as plt


def basis(x_vals, degree, h):
    mat = np.ones((x_vals.shape[-1], degree + 1))
    for i in range(0, degree):
        mat[:, i+1] = (mat[:, i] * x_vals)

    # we're dealing with the transpose of the monomial matrix
    np.transpose(mat)

    # we divide each index by h except first column
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
    h_vals = np.linspace(1, .0001, 50)
    error = []
    for j in range(0, len(h_vals)):
        n = 4  # number of pts to evaluate. we are told to evaluate  ˆx, ˆx+h, ˆx+2h, ˆx+3h, which is 4 pts
        h = h_vals[j]

        x_vals = np.linspace(0, 2 * math.pi, n+1)

        # calculating x pts based on x-hats given
        for i in range(0, n):
            x_vals[i] + (h * i)

        y_vals = f(x_vals)
        basis_primes = np.zeros(n + 1)
        basis_primes[2] = 2/h
        weights = find_weights(x_vals, basis_primes, h)

        approximation_sum = 0
        for k in range(0, len(weights)):
            approximation_sum = weights[k] * y_vals[k]

        f_double_prime = np.diff(np.diff(f(x_vals)))
        error.append(f_double_prime[0] - approximation_sum)  # calculating error for this particular h value and
        # appending it to the list of our errors

    plt.plot(h_vals, error, linewidth=4, label='Error')
    plt.xlabel('h')
    plt.ylabel('error')
    plt.legend()
    plt.title("Error as h -> 0")
    plt.show()


if __name__ == "__main__":
    main()
