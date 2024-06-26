from typing import Literal
import pandas as pd
import numpy as np
from numba import njit

# Simple Moving Average
def sma(source: pd.Series, length: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA)
    of the input time series data.

    Parameters:
    -----------
    source : pd.Series
        The time series data to calculate the SMA for.
    length : int
        The number of periods to include in the SMA calculation.

    Returns:
    --------
    pd.Series
        The calculated SMA time series data.
    """
    if len(source) < length:
        return pd.Series([], dtype=np.float64)
    
    sma_series = source.rolling(length).mean()
    return sma_series.dropna()

# Exponential Moving Average using Numba
@njit
def ema_numba(source: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate the Exponential Moving Average (EMA) using Numba.

    Parameters:
    -----------
    source : np.ndarray
        The input array of values.
    length : int
        The number of periods to include in the EMA calculation.

    Returns:
    --------
    np.ndarray
        The calculated EMA values.
    """
    alpha = 2 / (length + 1)
    ema_values = np.empty_like(source)
    ema_values[0] = source[0]
    for i in range(1, len(source)):
        ema_values[i] = alpha * source[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values

def ema(source: pd.Series, length: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA)
    of the input time series data.

    Parameters:
    -----------
    source : pd.Series
        The time series data to calculate the EMA for.
    length : int
        The number of periods to include in the EMA calculation.

    Returns:
    --------
    pd.Series
        The calculated EMA time series data.
    """
    if len(source) < length:
        return pd.Series([], dtype=np.float64)
    
    ema_values = ema_numba(source.to_numpy(), length)
    return pd.Series(ema_values, index=source.index).dropna()

# Smoothed Exponential Moving Average
def sema(source: pd.Series, length: int, smooth: int) -> pd.Series:
    """
    Calculate the Smoothed Exponential Moving Average (SEMA)
    of the input time series data.

    Parameters:
    -----------
    source : pd.Series
        The time series data to calculate the SEMA for.
    length : int
        The number of periods to include in the SEMA calculation.
    smooth : int
        The number of EMAs to smooth.

    Returns:
    --------
    pd.Series
        The calculated SEMA time series data.
    """
    if len(source) < length:
        return pd.Series([], dtype=np.float64)
    
    emas_dict = {}
    emas_dict["source_1"] = ema(source, length)
    for value in range(2, smooth + 1):
        emas_dict[f"source_{value}"] = ema(
            emas_dict[f"source_{value-1}"],
            length,
        )
    emas_df = pd.DataFrame(emas_dict)
    emas_df["sema"] = (
        emas_df[emas_df.columns[:-1]].diff(axis=1).sum(axis=1) * -1
        * smooth
        + emas_df[emas_df.columns[-1]]
    )
    sema_series = emas_df["sema"]
    return sema_series.dropna()

# Relative Moving Average using Pandas
def _rma_pandas(
    source: pd.Series,
    length: int,
    **kwargs
) -> pd.Series:
    """
    Calculate the Relative Moving Average (RMA) of the input time series
    data using Pandas.

    Parameters:
    -----------
    source : pd.Series
        The time series data to calculate the RMA for.
    length : int
        The number of periods to include in the RMA calculation.
    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to the pandas EWM (Exponential
        Weighted Moving Average) function.

    Returns:
    --------
    pd.Series
        The calculated RMA time series data.

    Note:
    -----
    The first values are different from the TradingView RMA.
    """
    if len(source) < length:
        return pd.Series([], dtype=np.float64)
    
    sma_series = (
        source
        .rolling(window=length, min_periods=length)
        .mean()[:length]
    )

    rest = source[length:]

    return (
        pd.concat([sma_series, rest])
        .ewm(alpha=1 / length, **kwargs)
        .mean()
    ).rename("RMA")

# Relative Moving Average using Numba
@njit
def _rma_numba(source: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate the Relative Moving Average (RMA) of the input time series
    data using Numba.

    Parameters:
    -----------
    source : np.ndarray
        The input array of values.
    length : int
        The number of periods to include in the RMA calculation.

    Returns:
    --------
    np.ndarray
        The calculated RMA values.
    """
    alpha = 1 / length
    rma_values = np.empty_like(source)
    rma_values[0] = np.mean(source[:length])
    for i in range(1, len(source)):
        rma_values[i] = alpha * source[i] + (1 - alpha) * rma_values[i - 1]
    return rma_values

def rma(
    source: pd.Series,
    length: int,
    method: Literal["numba", "pandas"] = "numba"
) -> pd.Series:
    """
    Calculate the Relative Moving Average (RMA) of the input time series
    data.

    Parameters:
    -----------
    source : pd.Series
        The time series data to calculate the RMA for.
    length : int
        The number of periods to include in the RMA calculation.
    method : {"numba", "pandas"}, optional
        The method to use for calculating the RMA, by default "numba".

    Returns:
    --------
    pd.Series
        The calculated RMA time series data.
    """
    if len(source) < length:
        return pd.Series([], dtype=np.float64)
    
    match method:
        case "numba":
            rma_values = _rma_numba(source.to_numpy(), length)
            return pd.Series(rma_values, index=source.index).dropna()
        case "pandas":
            return _rma_pandas(source, length)
        case _:
            raise TypeError("method must be 'numba' or 'pandas'")
