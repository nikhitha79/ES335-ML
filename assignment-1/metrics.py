from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    # Ensure the Series have the same index
    y_hat = y_hat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    acc = (y_hat == y).sum()/y.size
    return acc
    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """

    assert y_hat.size == y.size
    TP = ((y_hat == cls) & (y == cls)).sum()
    FP = ((y_hat == cls) & (y != cls)).sum()
    if TP + FP == 0:
        return 0.0
    prec = TP / (TP + FP)
    return prec
    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """

    assert y_hat.size == y.size
    TP = ((y_hat == cls) & (y == cls)).sum()
    FN = ((y_hat != cls) & (y == cls)).sum()
    if TP + FN == 0:
        return 0.0
    Rec = TP / (TP + FN)
    return Rec
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    assert y_hat.size == y.size
    rmse_val = np.sqrt(((y_hat - y) ** 2).mean())
    return rmse_val
    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    mae_val = (np.abs(y_hat - y)).mean()
    return mae_val
    pass
