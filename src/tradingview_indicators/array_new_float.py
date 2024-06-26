import numpy as np

def array_new_float(size=0, initial_value=np.nan):
    """
    Create a new array of float type elements.
    Parameters:
    size (int): Initial size of the array. Defaults to 0.
    initial_value (float): Initial value of all array elements. Defaults to NaN.
    Returns:
    np.ndarray: An array of float type elements.
    """
    if not isinstance(initial_value, (int, float)):
        raise ValueError("initial_value must be a float or an int")
    return np.full(size, initial_value, dtype=float)