import numpy as np
from numba import njit

@njit
def array_new_float(size=0, initial_value=np.nan):
    """
    Create a new array of float type elements.
    
    Parameters:
    -----------
    size : int
        Initial size of the array. Defaults to 0.
    initial_value : float
        Initial value of all array elements. Defaults to NaN.
    
    Returns:
    --------
    np.ndarray
        An array of float type elements.
    """
    return np.full(size, initial_value, dtype=np.float64)
