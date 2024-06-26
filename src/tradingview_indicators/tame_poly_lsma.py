import numpy as np
from numba import njit

@njit(cache=True, nogil=True)
def precompute_s(order):
    """
    Precompute the sigma approximation values for the given order.
    
    Parameters:
    -----------
    order : int
        The order for precomputing sigma values.
    
    Returns:
    --------
    np.ndarray
        Array of precomputed sigma values.
    """
    pi = 2 * np.arcsin(1)
    s_values = np.zeros(order)
    for i in range(order):
        x = ((i + 1) * pi) / order
        s_values[i] = np.sin(x) / x
    return s_values

@njit(cache=True, nogil=True)
def kernel(x, order, s_values):
    """
    Compute the kernel polynomial for given x and order.
    
    Parameters:
    -----------
    x : float
        The current x value.
    order : int
        The order for kernel computation.
    s_values : np.ndarray
        Precomputed sigma values.
    
    Returns:
    --------
    float
        The polynomial result.
    """
    pi = 2 * np.arcsin(1)
    b = 0.0
    for i in range(order):
        b += s_values[i] * np.sin(x * (i + 1) * pi) / (i + 1)
    pol = x * x + b
    return pol

@njit(cache=True, nogil=True)
def tame_poly_lsma(src, length, order):
    """
    Calculate the Tame Polynomial Least Squares Moving Average (LSMA).
    
    Parameters:
    -----------
    src : np.ndarray
        Source array of values.
    length : int
        The period length for the LSMA calculation.
    order : int
        The order for the polynomial estimation.
    
    Returns:
    --------
    np.ndarray
        The calculated LSMA values.
    """
    if length > len(src):
        raise ValueError("Length cannot be greater than length of source")
    if length < 1 or order < 1:
        raise ValueError("Length and order must be positive integers")
    
    s_values = precompute_s(order)
    result = np.zeros(len(src))
    
    for i in range(len(src)):
        if i < length:
            result[i] = np.nan  # Not enough data to compute the value
        else:
            sum_w = 0.0
            for j in range(length):
                sum_w += (kernel((j + 1) / length, order, s_values) - kernel(j / length, order, s_values)) * src[i - length + j]
            result[i] = sum_w
    
    return result

# Example usage with inputs