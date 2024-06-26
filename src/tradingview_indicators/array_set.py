import numpy as np

def array_set(arr, index, value):
    """
    Set the value of the element at the specified index.
    Parameters:
    arr (np.ndarray): The array object.
    index (int): The index of the element to be modified.
    value (float): The new value to be set.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("arr must be a numpy ndarray")
    if index < 0 or index >= arr.size:
        raise IndexError("index out of bounds")
    arr[index] = value