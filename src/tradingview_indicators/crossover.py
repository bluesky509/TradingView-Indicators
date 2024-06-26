import pandas as pd

def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect crossovers between two series.

    Parameters:
    -----------
    series1 : pd.Series
        The first series of values.
    series2 : pd.Series
        The second series of values.

    Returns:
    --------
    pd.Series
        A Series with boolean values indicating where crossovers occur.
    """
    if not isinstance(series1, pd.Series):
        raise ValueError("series1 must be a pandas Series")
    if not isinstance(series2, pd.Series):
        raise ValueError("series2 must be a pandas Series")
    if len(series1) != len(series2):
        raise ValueError("series1 and series2 must be of the same length")

    crossover = (series1.shift(1) < series2.shift(1)) & (series1 > series2)
    return crossover.astype(int).rename("Crossover")