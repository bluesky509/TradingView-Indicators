import numpy as np
import pandas as pd
from numba import njit

@njit
def wma_numba(values, length):
    """
    Calculate the Weighted Moving Average (WMA) using Numba.

    Parameters:
    values (np.ndarray): The input data array.
    length (int): The period of the WMA.

    Returns:
    np.ndarray: The WMA values.
    """
    weights = np.arange(1, length + 1)
    wma_values = np.empty_like(values)
    wma_values[:] = np.nan

    for i in range(length - 1, len(values)):
        wma_values[i] = np.dot(values[i-length+1:i+1], weights) / weights.sum()
    
    return wma_values

@njit
def hma_numba(values, length):
    """
    Calculate the Hull Moving Average (HMA) using Numba.

    Parameters:
    values (np.ndarray): The input data array.
    length (int): The period of the HMA.

    Returns:
    np.ndarray: The HMA values.
    """
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    wma_half = wma_numba(values, half_length)
    wma_full = wma_numba(values, length)
    raw_hma = 2 * wma_half - wma_full
    
    hma_values = wma_numba(raw_hma, sqrt_length)
    return hma_values

def hma(source: pd.Series, length: int) -> pd.Series:
    """
    Calculate the Hull Moving Average (HMA).

    Parameters:
    source (pd.Series): The input data.
    length (int): The period of the HMA.

    Returns:
    pd.Series: The HMA values.
    """
    hma_values = hma_numba(source.to_numpy(), length)
    return pd.Series(hma_values, index=source.index)