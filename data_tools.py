""" Module containing miscellaneous data tools"""

import numpy as np
import pandas as pd

import math


def rss(y) -> pd.Series:
    """ Returns the residual sum of squares of a dataset
    """
    return np.sum((y - np.mean(y)) ** 2)


def double_rss_sum(left, right):
    """ Returns sum of two residual sum of squares.
    """
    return (rss(left) + rss(right)).iloc[0]


def r_squared(y: pd.Series, y_hat: pd.Series) -> float:
    """Returns the r-squared value between actual and predicted values."""
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()
    return 1 - (ss_res/ss_tot)


def std_agg(cnt: int, s1: float, s2: float) -> float:
    """Returns standard deviation"""
    return math.sqrt((s2/cnt) - (s1/cnt)**2)
