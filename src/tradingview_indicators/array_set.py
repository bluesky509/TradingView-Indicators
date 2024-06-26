import numpy as np
from numba import njit

@njit
def array_set(arr, index, value):
    """
    Set the value of the element at the specified index.
    
    Parameters:
    -----------
    arr : np.ndarray
        The array object.
    index : int
        The index of the element to be modified.
    value : float
        The new value to be set.
    """
    arr[index] = value
