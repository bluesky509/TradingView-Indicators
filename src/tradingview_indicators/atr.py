import pandas as pd

def atr(high, low, close, length):
    """
    Calculate the Average True Range (ATR).

    Parameters:
    high (pd.Series): The high prices.
    low (pd.Series): The low prices.
    close (pd.Series): The close prices.
    length (int): The period of the ATR.

    Returns:
    pd.Series: The ATR values.
    """
    true_range = pd.DataFrame({
        'high_low': high - low,
        'high_close': (high - close.shift()).abs(),
        'low_close': (low - close.shift()).abs()
    }).max(axis=1)
    return true_range.rolling(window=length).mean()