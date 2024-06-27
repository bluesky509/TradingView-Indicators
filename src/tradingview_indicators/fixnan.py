import pandas as pd

def fixnan(source):
    """
    Replace NaN values with the previous nearest non-NaN value.

    Parameters:
    source (pd.Series): The input data.

    Returns:
    pd.Series: The input data with NaN values replaced by the previous nearest non-NaN value.
    """
    return source.ffill()
