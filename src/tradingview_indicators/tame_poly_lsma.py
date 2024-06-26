import numpy as np
import pandas as pd
from .array_new_float import array_new_float
from .array_set import array_set
from .array_sum import array_sum
from .math_sin import math_sin

# Sigma approximation function
def s(i, h):
    x = i * np.pi / h
    if x == 0:
        return 1
    else:
        return math_sin(x) / x

# Kernel computation function
def kernel(x, order):
    b = array_new_float(order)
    for i in range(1, order + 1):
        array_set(b, i - 1, s(i, order) * math_sin(x * i * np.pi) / i)
    pol = x * x + array_sum(b)
    return pol

# Main Polynomial LSMA Function
def tame_poly_lsma(src, length, order):
    if length > len(src):
        raise ValueError("Length cannot be greater than length of source")

    w = array_new_float(length)
    for i in range(1, length + 1):
        array_set(w, i - 1, (kernel(i / length, order) - kernel((i - 1) / length, order)) * src.iloc[i - 1])
    sum_w = array_sum(w)
    return sum_w