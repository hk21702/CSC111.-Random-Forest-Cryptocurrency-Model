""""
This module contains the custom data types WindowArgs, ForestArgs, TreeArgs, and DataPair
"""
from __future__ import annotations
import copy
from typing import Union
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class WindowArgs():
    """Class containing the basic arguments used for buildings window inputs for a
    model.

        Instance attributes:
        - window_size: The size of each input window.
        - target_shift: The number of days out from the last day of the window
        the model is trying to predict for.
        - req_features: A set with the symbol or search terms required by the window.
        - data_frames: List of dataframes being used to construct the windows.
        - target: Target data for supervised learning.
        - target_lbl: Target feature label.
    """
    window_size: int
    target_shift: int
    req_features: set[str]
    data_frames: list[pd.DataFrame]
    target: pd.DataFrame
    target_lbl: str


@dataclass
class ForestArgs():
    """Class containing the basic arguments used for building a random forest.

        Instance attributes:
        - n_trees: The number of uncorrelated trees in the forest.
        - sample_sz: The number of rows selected and passed to each tree of the forest.
        - max_depth: The maximum depth of each decision tree.
        - min_leaf: The minimum number of rows required for a tree node to continue splitting.
        - seed: Numpy random seed.
        - test_percentage: The percentage of the dataset to be used for testing.
        - n_features: The number of features selected and passed to each tree. Either an integer,
        or 'sqrt', the square root of all avaliable features, or 'log2', the log2 of all avaliable
        features.

        Representation Invariants:
        - n_trees > 0
        - sample_sz > 0
        - min_leaf > 0
        - 0 < self.test_percentage < 1
        - self.n_features in {'sqrt', 'log2'} or type(self.n_features) == int
    """
    n_trees: int
    sample_sz: int
    max_depth: int = 10
    min_leaf: int = 3
    seed: int = 12
    test_percentage: float = 0.10
    n_features: Union[str, int] = 'sqrt'


@dataclass
class TreeArgs():
    """Class with the basic arguments used for a decision tree.

        Instance attributes:
        - n_features: The number of features being used for the node.
        - idxs: The indicies of the rows the node contains.
        - f_idxs: The features the node contains.
        - train: DataPair of the x and y training sets.
        - depth: The remaining depth this decision tree.
        - min_leaf: The minimum number of rows required for a tree node to continue splitting.
    """
    n_features: int
    idxs: np.ndarray
    f_idxs: np.ndarray
    train: DataPair
    depth: int = 10
    min_leaf: int = 5

    def split(self, f_idxs: np.ndarray, idxs: np.ndarray) -> TreeArgs:
        """Returns a new TreeParams with modified attributes for use for
        a child node."""
        new_params = copy.copy(self)
        new_params.depth -= 1
        new_params.idxs = new_params.idxs[idxs]
        new_params.f_idxs = f_idxs
        return new_params


@dataclass
class DataPair():
    """Class which contains the x, y pair for a given dataset.

        Instance attributes:
        - x: Independent variables of training set.
        - y: Depdendent variables of training set.
    """
    x: pd.DataFrame
    y: pd.DataFrame


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['numpy', 'pandas', 'data_tools', 'tqdm.auto',
                          'data_ingest', 'dataclasses', 'copy'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E1136']
    })
