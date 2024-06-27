import pandas as pd

def cum(source):
    """
    Calculate the cumulative sum of the source.

    Parameters:
    source (pd.Series): The input data.

    Returns:
    pd.Series: The cumulative sum values.
    """
    return source.cumsum()