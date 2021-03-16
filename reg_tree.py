""" Module containing RegTree class being used as the
regression tree datatype to store the conditions and structure
of the regression tree model.
"""
from __future__ import annotations
from typing import Optional


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
    condition: str

    # Private Instance Attributes:
    # - _left TODO
    # - _right TODO
    _left: RegTree
    _right: RegTree

    def __init__(self, condition: str, _left: RegTree  = None, 
    _right: RegTree = None) -> None:
        """ Initialize a new regression tree.
        
        """
        self.condition = condition
        self._left = _left
        self._right = _right