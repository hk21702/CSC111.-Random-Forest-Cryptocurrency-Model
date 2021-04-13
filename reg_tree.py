""" Module containing RegTree class being used as the
regression tree datatype to store the conditions and
structure of the regression tree model. Also includes
relevant supporting custom exceptions and errors.
"""
from __future__ import annotations
import os
from os.path import split
import psutil
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pandas as pd
import data_tools
import training
import data_ingest
from dataclasses import dataclass
from typing import Optional


class MissingPrediction(Exception):
    """ Exception raised when there are no valid subnodes
        and the current leaf has no prediction."""


class InvalidFeatures(Exception):
    """ Exception raised when the sample featureset
    is incompatible with the current model."""


class RegTree:
    """ A regression tree for the cryptocurrency
    prediction model.

    Instance attributes:
        - TODO
        - TODO
        - TODO

    Representation Invariants:
        - TODO
    """
    feature: str
    threshold: float
    prediction: Optional[float]

    # Private Instance Attributes:
    # - _left TODO
    # - _right TODO
    _left: Optional[RegTree]
    _right: Optional[RegTree]

    def __init__(self, _left: RegTree = None,
                 _right: RegTree = None) -> None:
        """ Initialize a new regression tree.

        """
        self._left = _left
        self._right = _right

        self.feature = None
        self.threshold = None
        self.prediction = None

    def split(self, main_process_id: int, x_train, y_train, depth: int, max_depth: int) -> None:
        """ Split into two nodes based on a given rule.
        If final depth is reached or there is not enough
        training data to make rules, set prediction output.
        """
        if depth == max_depth or len(x_train) < 2:
            self._prediction = np.mean(y_train)
        else:
            num_cores = multiprocessing.cpu_count()

            rule = training.get_best_rule(x_train, y_train)
            left_split = x_train[rule['feature']] < rule['threshold']
            self.feature = rule['feature']
            self.threshold = rule['threshold']

            main_process = psutil.Process(main_process_id)
            children = main_process.children(recursive=True)
            active_processes = len(children) + 1  # plus parent
            new_processes = 2

            if num_cores >= active_processes + new_processes:
                pool = MyPool(num_cores)
                input_params = [('left', main_process_id,
                                 x_train[left_split], y_train[left_split], depth + 1, max_depth),
                                ('right', main_process_id,
                                 x_train[~left_split], y_train[~left_split], depth + 1, max_depth)]
                results = pool.starmap(self.create_branch, input_params)
                pool.close()
                pool.join()
            else:  # serial
                self._left = RegTree().split(main_process_id,
                                             x_train[left_split], y_train[left_split], depth + 1, max_depth)
                self._right = RegTree().split(main_process_id,
                                              x_train[~left_split], y_train[~left_split], depth + 1, max_depth)

    def create_branch(self, side: str, main_process_id: int, x_train, y_train, depth: int, max_depth: int):
        if side == 'left':
            self._left = RegTree().split(main_process_id, x_train, y_train, depth, max_depth)
        if side == 'right':
            self._right = RegTree().split(main_process_id, x_train, y_train, depth, max_depth)


class Model:
    """ Class for keeping track of a regression
    tree's usage requirements.
    """
    tree: RegTree
    required_features: set[str]
    window_size: int

    def __init__(self, window_size: int = None) -> None:
        """ Initialize a new model.

        """
        self.tree = RegTree()
        self.window_size = window_size
        self.required_features = set()

    def predict(self, sample) -> float:
        """ Returns a prediction based on input timeline. Raises
        InvalidFeatures if the given sample has improper features"""
        current_node = self.tree
        prediction = None
        try:
            while prediction is None:
                if sample[current_node.feature] < current_node.threshold:
                    current_node = current_node._left
                else:
                    current_node = current_node._right
                prediction = current_node.prediction

                if current_node._left is None and prediction is None:
                    raise MissingPrediction
            return prediction
        except KeyError:
            raise InvalidFeatures

    def train(self, features: list[pd.DataFrame], target: pd.Series, target_shift: int, max_depth: int, window_size: int) -> None:
        self.window_size = window_size
        x_train, y_train = data_ingest.create_training_input(
            self.window_size, features, target, target_shift)
        print('created inputs')
        process_id = os.getpid()
        self.tree.split(process_id, x_train, y_train, 0, max_depth)
        self.required_features = set(x_train.columns)

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        predictions = x_test.apply(self.predict, axis='columns')
        return data_tools.r_squared(y_test, predictions)


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
