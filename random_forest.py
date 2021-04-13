""" Module containing RandomForest class being used as the
random forest datatype to store the conditions and
structure of the regression tree model. Also includes the
DecisionTree class which are the decision tree regressors
used in the random forest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import data_ingest
from data_tools import std_agg, r_squared
from data_classes import TreeArgs, WindowArgs, ForestArgs, DataSet


class RandomForest():
    """ A random forest regressor for the cryptocurrency
    prediction model.

    Instance attributes:
        - forest: ForestArg containing the arguments used to build the
        random forest regressor.
        - window: WindowArgs containing the arguments used to build the
        windowed inputs used for training and testing.
        - requirements: The required features to use the random forest model.
    """
    forest: ForestArgs
    window: WindowArgs
    requirements: pd.core.indexes.base.Index

    # Private Instance Attributes:
    # - _train: DataSet of the training set variables.
    # - _test: Data set of the testing set variables.
    # - _n_features: The number of features being used to build each tree.
    # - _trees: List of the decision trees of the random forest.
    _train: DataSet
    _test: DataSet
    _n_features: int
    _trees: list[DecisionTree] = []

    def __init__(self, window: WindowArgs, forest: ForestArgs,

                 ) -> None:
        self.forest = forest
        self.window = window

        np.random.seed(forest.seed)

        self._train = data_ingest.create_training_input(
            window)

        print('Created training inputs')

        test_idxs = np.random.permutation(
            len(self._train.y))[:int(self._train.x.shape[0] * forest.test_percentage)]

        self._test = DataSet(self._train.x.iloc[test_idxs],
                             self._train.y.iloc[test_idxs].squeeze())
        self._train.x = self._train.x.drop(test_idxs)
        self._train.y = self._train.y.drop(test_idxs)

        self.requirements = self._train.x.columns

        if forest.n_features == 'sqrt':
            self._n_features = int(np.sqrt(self._train.x.shape[1]))
        elif forest.n_features == 'log2':
            self._n_features = int(np.log2(self._train.x.shape[1]))
        elif forest.n_features > self._train.x.shape[0]:
            print("Manual number of features too large. "
                  f"Setting to max size: {self._train.x.shape[1]}")
            self._n_features = self._train.x.shape[1]
        else:
            self._n_features = forest.n_features
        print(
            f'Using {self._n_features}, out of, {self._train.x.shape[1]} features')

        if self._train.x.shape[1] - 2 < self.forest.sample_sz:
            print(
                f'Sample size too large. Setting to max size: {self._train.y.shape[0]}')
            self.forest.sample_sz = self._train.y.shape[0]

        for _ in tqdm(range(forest.n_trees)):
            self._trees.append(self.create_tree())
            print(f'\n R-Squared: {self.accuracy}')

    def create_tree(self) -> DecisionTree:
        """Returns a new decision tree."""
        idxs = np.random.permutation(len(self._train.y))[
            :self.forest.sample_sz]
        f_idxs = np.random.permutation(self._train.x.columns)[
            :self._n_features]

        train = DataSet(self._train.x.iloc[idxs], self._train.y.iloc[idxs])

        tree_params = TreeArgs(self._n_features,
                               np.array(range(self.forest.sample_sz)),
                               f_idxs, train,
                               self.forest.max_depth, self.forest.min_leaf)
        return DecisionTree(tree_params)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns an numpy array of predictions from each row of inputs.
        Averages the predictions made by each tree of the forest."""
        return np.mean([t.predict(x) for t in self._trees], axis=0)

    @property
    def accuracy(self) -> float:
        """Gets the RSquared value using the testing dataset."""
        results = self.predict(self._test.x)
        return r_squared(self._test.y, results)


class DecisionTree():
    """ A decision tree regressor for the cryptocurrency
    prediction model.

    Instance attributes:
        - args: TreeParams containing the arguments used to build the tree.
        - val: The prediction of a given node. Is the average of avaliable observations
        if the node is not a leaf.
        - var_idx: Feature name for which the split to any children nodes on based on.
        - split: Value threshold for which the split to any children nodes are based on.
    """
    args: TreeArgs
    val: np.float64
    var_idx: str = None
    split: np.float64 = None

    # Private Instance Attributes:
    # - _left: The left child node.
    # - _right: The right child node.
    # - _score: Current split condition rule score.
    _left: DecisionTree
    _right: DecisionTree
    _score: float

    def __init__(self, params: TreeArgs) -> None:
        self.args = params
        self.val = np.mean(self.args.train.y.iloc[self.args.idxs].values)
        self._score = float('inf')
        self.find_varsplit()

    def find_varsplit(self) -> None:
        """Finds the best avaliable split and creates
        the accompanying children nodes for the tree."""
        for i in self.args.f_idxs:
            self.find_better_split(i)
        if self.is_leaf:
            return
        x = self._split_col
        left = np.nonzero(x <= self.split)[0]
        right = np.nonzero(x > self.split)[0]
        lf_idxs = np.random.permutation(self.args.train.x.columns)[
            :self.args.n_features]
        rf_idxs = np.random.permutation(self.args.train.x.columns)[
            :self.args.n_features]
        self._left = DecisionTree(self.args.split(lf_idxs, left))
        self._right = DecisionTree(self.args.split(rf_idxs, right))

    def find_better_split(self, var_idx: str) -> None:
        """Finds the best avaliable split thresholds for a given column."""
        x = self.args.train.x[var_idx].values[self.args.idxs]
        y = self.args.train.y.values[self.args.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]

        self._split(sort_x, sort_y, var_idx)

    def _split(self, sort_x: np.ndarray, sort_y: np.ndarray, var_idx: str) -> None:
        """Helper function which uses a brute force algorithm to find the best
        avaliable split thresholds for a given column."""
        right_cnt, right_sum, right_sum2 = len(
            self.args.idxs), sort_y.sum(), (sort_y**2).sum()
        left_cnt, left_sum, left_sum2 = 0, 0., 0.

        for i in range(0, len(self.args.idxs) - self.args.min_leaf - 1):
            yi = sort_y[i]
            left_cnt += 1
            right_cnt -= 1
            left_sum += yi
            right_sum -= yi
            left_sum2 += yi**2
            right_sum2 -= yi**2
            if i < self.args.min_leaf or sort_x[i] == sort_x[i + 1]:
                continue

            left_std = std_agg(left_cnt, left_sum, left_sum2)
            right_std = std_agg(right_cnt, right_sum, right_sum2)
            curr_score = left_std * left_cnt + right_std * right_cnt
            if curr_score < self._score:
                self.var_idx, self._score, self.split = var_idx, curr_score, sort_x[i]

    @property
    def _split_col(self) -> np.ndarray:
        "Gets the column at self.var_idx with elements at self.args.idxs"
        return self.args.train.x[self.var_idx].values[self.args.idxs]

    @property
    def is_leaf(self) -> bool:
        """Gets a boolean representing whenever or not the current node is a leaf."""
        return self._score == float('inf') or self.args.depth <= 0

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns an numpy array of predictions from each row of inputs."""
        return np.array([self._predict_row(xi[1]) for xi in x.iterrows()])

    def _predict_row(self, xi: pd.Series) -> any:
        """Returns a prediction based on a row of inputs."""
        if self.is_leaf:
            return self.val
        t = self._left if xi[self.var_idx] <= self.split else self._right
        return t._predict_row(xi)


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
