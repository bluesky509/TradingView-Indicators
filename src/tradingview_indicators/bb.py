import pandas as pd

def bb(source, length, mult):
    """
    Calculate Bollinger Bands.

    Parameters:
    source (pd.Series): The input data.
    length (int): The period of the BB.
    mult (float): The multiplier for the standard deviation.

    Returns:
    tuple: A tuple containing the middle band, upper band, and lower band.
    """
    basis = source.rolling(window=length).mean()
    dev = mult * source.rolling(window=length).std()
    upper_band = basis + dev
    lower_band = basis - dev
    return basis, upper_band, lower_band