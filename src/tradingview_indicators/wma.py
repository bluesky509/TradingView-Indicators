import pandas as pd

def wma(series: pd.Series, length: int) -> pd.Series:
    """
    Calculate the Weighted Moving Average (WMA) for the given series.

    Parameters:
    -----------
    series : pd.Series
        The input series of values.
    length : int
        The period for calculating the WMA.

    Returns:
    --------
    pd.Series
        The Weighted Moving Average (WMA) values.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("series must be a pandas Series")
    if length < 1:
        raise ValueError("length must be a positive integer")

    # Compute weights
    weights = pd.Series(range(1, length + 1))
    
    def calc_wma(window):
        return (window * weights).sum() / weights.sum()

    wma = series.rolling(window=length).apply(calc_wma, raw=False)
    return wma.rename("WMA")