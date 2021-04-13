from __future__ import annotations
from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import data_ingest
from data_tools import std_agg, r_squared
from data_classes import WindowParams, ForestParams, DataSet


class RandomForest():
    """ A random forest regressor for the cryptocurrency
    prediction model.

    Instance attributes:
        - n_trees: The number of decision trees in the random forest.
        - n_features: The number of features sampled and passed to each
        tree: 'sqrt' or 'log2' or integer
        - sample_sz: The number of training samples selected and passed to each tree.
        - depth: The max depth of each decision tree.
        - min_leaf: The minimum number of sample rows needed for further splitting of a node.

    Representation Invariants:
    """
    forest_params: ForestParams
    min_leaf: int

    # Private Instance Attributes:
    # - _train: DataSet of the training set variables.
    # - _test: Data set of the testing set variables.
    # - _trees: List of the decision trees of the random forest.
    _train: DataSet
    _test: DataSet
    _n_features: Union(str, int)
    _trees: list[DecisionTree] = []

    def __init__(self, window: WindowParams, params: ForestParams,
                 depth: int = 10, min_leaf: int = 5,
                 ) -> None:
        self.forest_params = params

        np.random.seed(params.seed)

        self._train = data_ingest.create_training_input(
            window)

        print('created inputs')

        self.depth, self.min_leaf = depth, min_leaf

        test_idxs = np.random.permutation(
            len(self._train.y))[:int(self._train.x.shape[0] * params.test_percentage)]

        self._test = DataSet(self._train.x.iloc[test_idxs],
                             self._train.y.iloc[test_idxs].squeeze())
        self._train.x = self._train.x.drop(test_idxs)
        self._train.y = self._train.y.drop(test_idxs)

        self.requirements = self._train.x.columns

        if params.n_features == 'sqrt':
            self._n_features = int(np.sqrt(self._train.x.shape[1]))
        elif params.n_features == 'log2':
            self._n_features = int(np.log2(self._train.x.shape[1]))
        elif params.n_features > self._train.x.shape[0]:
            print("Manual number of features too large. "
                  f"Setting to max size: {self._train.x.shape[1]}")
            self._n_features = self._train.x.shape[1]
        else:
            self._n_features = params.n_features
        print(
            f'Using {self._n_features}, out of, {self._train.x.shape[1]} features')

        if self._train.x.shape[1] - 2 < self.forest_params.sample_sz:
            print(
                f'Sample size too large. Setting to max size: {self._train.y.shape[0]}')
            self.forest_params.sample_sz = self._train.y.shape[0]

        for _ in tqdm(range(params.n_trees)):
            self._trees.append(self.create_tree())
            print(f'R-Squared: {self.accuracy}')

    def create_tree(self) -> DecisionTree:
        idxs = np.random.permutation(len(self._train.y))[
            :self.forest_params.sample_sz]
        f_idxs = np.random.permutation(self._train.x.columns)[
            :self._n_features]
        return DecisionTree(DataSet(self._train.x.iloc[idxs], self._train.y.iloc[idxs]),
                            self._n_features, f_idxs,
                            idxs=np.array(range(self.forest_params.sample_sz)),
                            depth=self.depth, min_leaf=self.min_leaf)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([t.predict(x) for t in self._trees], axis=0)

    @property
    def accuracy(self) -> float:
        results = self.predict(self._test.x)
        return r_squared(self._test.y, results)


class DecisionTree():
    var_idx: str = None

    def __init__(self, train: DataSet, n_features: int,
                 f_idxs: np.ndarray, idxs: np.ndarray, depth: int = 10, min_leaf: int = 5) -> None:
        self.idxs, self.min_leaf, self.f_idxs = idxs, min_leaf, f_idxs
        self._train = train
        self.depth = depth
        self.n_features = n_features
        self.n, self.c = len(idxs), self._train.x.shape[1]
        self.val = np.mean(self._train.y.iloc[idxs].values)
        self._score = float('inf')
        self.find_varsplit()

    def find_varsplit(self) -> None:
        for i in self.f_idxs:
            self.find_better_split(i)
        if self.is_leaf:
            return
        x = self.split_col
        left = np.nonzero(x <= self.split)[0]
        right = np.nonzero(x > self.split)[0]
        lf_idxs = np.random.permutation(self._train.x.columns)[
            :self.n_features]
        rf_idxs = np.random.permutation(self._train.x.columns)[
            :self.n_features]
        self._left = DecisionTree(self._train, self.n_features, lf_idxs,
                                  self.idxs[left], depth=self.depth - 1, min_leaf=self.min_leaf)
        self._right = DecisionTree(self._train, self.n_features, rf_idxs,
                                   self.idxs[right], depth=self.depth - 1, min_leaf=self.min_leaf)

    def find_better_split(self, var_idx: str) -> None:
        x, y = self._train.x[var_idx].values[self.idxs], self._train.y.values[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        right_cnt, right_sum, right_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        left_cnt, left_sum, left_sum2 = 0, 0., 0.

        for i in range(0, self.n - self.min_leaf - 1):
            xi, yi = sort_x[i], sort_y[i]
            left_cnt += 1
            right_cnt -= 1
            left_sum += yi
            right_sum -= yi
            left_sum2 += yi**2
            right_sum2 -= yi**2
            if i < self.min_leaf or xi == sort_x[i + 1]:
                continue

            left_std = std_agg(left_cnt, left_sum, left_sum2)
            right_std = std_agg(right_cnt, right_sum, right_sum2)
            curr_score = left_std * left_cnt + right_std * right_cnt
            if curr_score < self._score:
                self.var_idx, self._score, self.split = var_idx, curr_score, xi

    @property
    def split_col(self) -> np.ndarray:
        return self._train.x[self.var_idx].values[self.idxs]

    @property
    def is_leaf(self) -> bool:
        return self._score == float('inf') or self.depth <= 0

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_row(xi[1]) for xi in x.iterrows()])

    def predict_row(self, xi: pd.Series) -> any:
        if self.is_leaf:
            return self.val
        t = self._left if xi[self.var_idx] <= self.split else self._right
        return t.predict_row(xi)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['numpy', 'pandas', 'data_tools', 'tqdm.auto',
                          'data_ingest', 'data_classes'],
        'allowed-io': ['__init__'],
        'max-line-length': 100,
        'disable': ['E1136']
    })
