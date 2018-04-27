import numpy as np


def f():
    A = np.zeros(3, dtype=float)
    B = np.zeros(3, dtype=float)
    A[2] = 5
    B[0] = 3
    A[1] = 1
    print(np.sum(np.multiply(A,B)))
    print(A+B)
    print(B + 3*A)
    print(A)
    print(B/3)

f()