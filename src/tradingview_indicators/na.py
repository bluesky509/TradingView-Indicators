import pandas as pd

def na(x):
    """
    Check if the value is NaN.

    Parameters:
    x (pd.Series or scalar): The value to be tested.

    Returns:
    bool: True if x is NaN, False otherwise.
    """
    return pd.isna(x)
