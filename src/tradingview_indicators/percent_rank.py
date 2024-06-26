import pandas as pd

def percentrank(source: pd.Series, length: int) -> pd.Series:
    """
    Calculate the percent rank.

    Parameters:
    ----------
    source : pd.Series
        The input data.
    length : int
        The period over which to calculate the percent rank.

    Returns:
    -------
    pd.Series
        The percent rank values.
    """
    def rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1] * 100
    
    return source.rolling(window=length).apply(rank, raw=False).rename("PercentRank")
