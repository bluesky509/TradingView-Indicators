import pandas as pd

def stoch(source, high, low, length) -> pd.Series:
    """
    Calculate the Fast Stochastic Oscillator values for the given
    period length.

    Parameters:
    -----------
    length : int
        The length of the stochastic period.

    Returns:
    --------
    pd.Series
        The Fast Stochastic Oscillator values.
    """
    lowest_low = low.rolling(length).min()
    hightest_high = high.rolling(length).max()
    stochastic = (
        100
        * (source - lowest_low)
        / (hightest_high - lowest_low)
    )
    return stochastic.rename("stoch")
