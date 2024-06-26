import pandas as pd

def zscore_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Z-Score of the Exponential Moving Average (EMA) for the given series.

    Parameters:
    -----------
    series : pd.Series
        The input series of values.
    period : int
        The period for calculating the EMA and standard deviation.

    Returns:
    --------
    pd.Series
        The Z-Score of the EMA values.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("series must be a pandas Series")
    if period < 1:
        raise ValueError("period must be a positive integer")

    ema = series.ewm(span=period, adjust=False).mean()
    stddev = series.rolling(window=period).std()
    zscore = (series - ema) / stddev
    return zscore.rename("ZScore_EMA")