import numpy as np

def math_sin(angle):
    """
    Calculate the trigonometric sine of an angle.
    Parameters:
    angle (float): The angle in radians.
    Returns:
    float: The sine of the angle.
    """
    if not isinstance(angle, (int, float)):
        raise ValueError("angle must be a float or an int")
    return np.sin(angle)
