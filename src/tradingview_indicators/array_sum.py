import numpy as np
from numba import njit

@njit
def array_sum(arr):
    """
    Return the sum of an array's elements.
    
    Parameters:
    -----------
    arr : np.ndarray
        The array object.
    
    Returns:
    --------
    float
        The sum of the array's elements.
    """
    return np.nansum(arr)
