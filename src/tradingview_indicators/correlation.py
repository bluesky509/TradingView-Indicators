import pandas as pd

def correlation(source1, source2, length):
    """
    Calculate the correlation coefficient.

    Parameters:
    source1 (pd.Series): The first input data.
    source2 (pd.Series): The second input data.
    length (int): The period over which to calculate the correlation.

    Returns:
    pd.Series: The correlation values.
    """
    return source1.rolling(window=length).corr(source2)