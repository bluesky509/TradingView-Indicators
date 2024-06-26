import numpy as np

def array_sum(arr):
    """
    Return the sum of an array's elements.
    Parameters:
    arr (np.ndarray): The array object.
    Returns:
    float: The sum of the array's elements.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("arr must be a numpy ndarray")
    return np.nansum(arr)