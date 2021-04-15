""" Module containing miscellaneous data tool functions"""

import math


import pandas as pd


def r_squared(y: pd.Series, y_hat: pd.Series) -> float:
    """Returns the r-squared value between actual and predicted values."""
    y_bar = y.mean()
    ss_tot = ((y - y_bar)**2).sum()
    ss_res = ((y - y_hat)**2).sum()
    return 1 - (ss_res / ss_tot)


def std_agg(cnt: int, s1: float, s2: float) -> float:
    """Returns standard deviation"""
    a = (s2 / cnt) - (s1 / cnt)**2
    if abs(a) < 0.00001:
        # Potential floating point arithmetic error (Or it really is 0).
        return 0
    else:
        return math.sqrt(a)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['pandas', 'math'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E1136']
    })
