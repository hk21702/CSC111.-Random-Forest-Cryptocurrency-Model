""""
This module contains the custom data types TODO
"""
from __future__ import annotations
import copy
from typing import Union

import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass
class WindowParams():
    """A dataclass which represents the basic paramasters used for buildings window inputs for a
    model."""
    window_size: int
    target_shift: int
    data_frames: list[pd.DataFrame]
    target: pd.DataFrame


@dataclass
class ForestParams():
    """A dataclass which represents the basic paramaters used for building a random forest.

       Representation Invariants:
       - 0 < self.test_percentage < 1
       - self.n_features in {'sqrt', 'log2'} or type(self.n_features) == int
       - n_trees > 0
       - sample_sz > 0
    """
    n_trees: int
    #tree_params: TreeParams
    sample_sz: int
    seed: int = 12
    test_percentage: float = 0.10
    n_features: Union[str, int] = 'sqrt'


@dataclass
class DataSet():
    """A dataclass which contains the x, y pair for a given dataset"""
    x: pd.DataFrame
    y: pd.DataFrame


@dataclass
class TreeParams():
    """ A dataclass which represents the basic paramaters used for a decision tree"""
    n_features: int
    idxs: np.ndarray
    f_idxs: np.ndarray
    _train: DataSet
    depth: int = 10
    min_leaf: int = 5

    def split(self, f_idxs: np.ndarray, idxs: np.ndarray) -> TreeParams:
        new_params = copy.copy(self)
        new_params.depth -= 1
        new_params.idxs = new_params.idxs[idxs]
        new_params.f_idxs = f_idxs
        return new_params

    # idxs:
    # f_idxs:
