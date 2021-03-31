""" Module containing miscellaneous data tools"""

import numpy as np


def rss(y) -> np.ndarray:
    """ Returns the residual sum of squares of a dataset
    """
    return np.sum((y - np.mean(y)) ** 2)


def double_rss_sum(left, right):
    """ Returns sum of two residual sum of squares.
    """
    return rss(left) + rss(right)
