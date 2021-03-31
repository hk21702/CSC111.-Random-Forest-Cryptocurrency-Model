""" Module containing RegTree class being used as the
regression tree datatype to store the conditions and
structure of the regression tree model. Also includes
relevant supporting custom exceptions and errors.
"""
from os.path import split
import numpy as np
import data_tools
import training
from dataclasses import dataclass
from __future__ import annotations
from typing import Optional


class MissingPrediction(Exception):
    """ Exception raised when there are no valid subnodes
        and the current leaf has no prediction."""


class InvalidFeatures(Exception):
    """ Exception raised when the sample featureset
    is incompatible with the current model."""


@dataclass
class Model:
    """ Class for keeping track of a regression
    tree's usage requirements.
    """
    tree: RegTree
    required_features: set[str]
    window_size: int

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

    def train(self, x_train, y_train, depth: int, max_depth: int) -> None:
        self.tree.split(x_train, y_train, depth, max_depth)
        self.required_features = set(x_train.columns)


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

    def split(self, x_train, y_train, depth: int, max_depth: int) -> None:
        """ Split into two nodes based on a given rule.
        If final depth is reached or there is not enough
        training data to make rules, set prediction output.
        """
        if depth == max_depth or len(x_train) < 2:
            self._prediction = np.mean(y_train)
        else:
            rule = training.get_best_rule(x_train, y_train)
            left_split = x_train[rule['feature']] < rule['threshhold']
            self.feature = rule['feature']
            self.threshold = rule['threshold']
            self._left = RegTree().split(
                x_train[left_split], y_train[left_split], depth + 1, max_depth)
            self._left = RegTree().split(
                x_train[~left_split], y_train[~left_split], depth + 1, max_depth)

