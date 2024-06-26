import pandas as pd
import numpy as np

def linreg(source: pd.Series, length: int, offset: int = 0) -> pd.Series:
    """
    Calculate the Linear Regression curve for the given series.

    Parameters:
    -----------
    source : pd.Series
        The input series of values.
    length : int
        The period for calculating the Linear Regression.
    offset : int, optional
        The offset to be applied in the formula (default is 0).

    Returns:
    --------
    pd.Series
        The Linear Regression curve values.
    """
    if not isinstance(source, pd.Series):
        raise ValueError("source must be a pandas Series")
    if length < 1:
        raise ValueError("length must be a positive integer")
    
    x = np.arange(length)
    x_mean = np.mean(x)
    
    def calc_slope(y):
        y_mean = np.mean(y)
        return np.dot(y - y_mean, x - x_mean) / np.dot(x - x_mean, x - x_mean)

    def calc_intercept(y):
        return np.mean(y) - calc_slope(y) * x_mean

    slope = source.rolling(window=length).apply(calc_slope, raw=False)
    intercept = source.rolling(window=length).apply(calc_intercept, raw=False)
    
    linreg = intercept + slope * (length - 1 - offset)
    return linreg.rename("LinReg")
