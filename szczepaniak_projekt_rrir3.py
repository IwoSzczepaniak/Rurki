import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad


# Preprocessing of the left side matrix of the system of equations
def X_preprocesing():
    X = np.zeros((n + 1, n + 1))
    for i in range(1, n):
        for j in range(n+1):
            f_1 = lambda x: de_i(i, x) * de_i(j, x)
            X[i, j] = -quad(f_1, elem_size * (i - 1), elem_size * (i + 1))[0]

    X[0, 0] = 1
    X[n, n] = 1
    return X


# Preprocessing of the right side matrix of the system of equations
def Y_preprocesing():
    Y = np.zeros(n + 1)
    for i in range(1, n):
        Y[i] = \
        quad(lambda x: 4 * math.pi * p(x) * e_i(i, x) + (2 / 3) * de_i(i, x), elem_size * (i - 1), elem_size * i)[0]
    Y[0] = 0
    Y[n] = 0
    return Y


# returns i-th point
def x_i(i):
    return elem_size * i


def p(x):
    if 1 < x <= 2:
        return 1
    else:
        return 0


def e_i(i, x):
    if x_i(i - 1) <= x < x_i(i):
        return (x - x_i(i - 1)) / elem_size
    elif x_i(i) <= x <= x_i(i + 1):
        return (x_i(i + 1) - x) / elem_size
    else:
        return 0.


def de_i(i, x):
    if x_i(i - 1) <= x < x_i(i):
        return 1 / elem_size
    elif x_i(i) <= x <= x_i(i + 1):
        return -1 / elem_size
    else:
        return 0.


def prepare_to_add_to_plot(x, a):
    value = 0
    for i in range(n):
        value += a[i] * e_i(i, x)
    return value + 5 + (2 / 3) * x


if __name__ == '__main__':
    # Discretize the domain into n elements
    n = 20
    elem_size = 3 / n

    X = X_preprocesing()
    Y = Y_preprocesing()

    # a = gaussian_elimination(X, Y)
    a = np.linalg.solve(X, Y)

    # # Draw the approximation
    x = np.linspace(0, 3, n)
    y = [prepare_to_add_to_plot(curr_x, a) for curr_x in x]
    plt.title("U function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y, 'go-', label='u')
    plt.legend()
    plt.show()
