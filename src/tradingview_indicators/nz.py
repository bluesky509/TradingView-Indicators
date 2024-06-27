import pandas as pd

def nz(source, replacement=0):
    """
    Replace NaN values with zeros or a given value.

    Parameters:
    source (pd.Series or scalar): The input data.
    replacement (scalar, optional): The value to replace NaN with. Defaults to 0.

    Returns:
    pd.Series or scalar: The input data with NaN values replaced.
    """
    return source.fillna(replacement)
