import pandas as pd

def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect crossunders between two series.

    Parameters:
    -----------
    series1 : pd.Series
        The first series of values.
    series2 : pd.Series
        The second series of values.

    Returns:
    --------
    pd.Series
        A Series with boolean values indicating where crossunders occur.
    """
    if not isinstance(series1, pd.Series):
        raise ValueError("series1 must be a pandas Series")
    if not isinstance(series2, pd.Series):
        raise ValueError("series2 must be a pandas Series")
    if len(series1) != len(series2):
        raise ValueError("series1 and series2 must be of the same length")

    crossunder = (series1.shift(1) > series2.shift(1)) & (series1 < series2)
    return crossunder.astype(int).rename("Crossunder")