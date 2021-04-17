"""Module containing PandasModel class for interfacing Qt view with a pandas dataframe. """
import pandas as pd

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex


class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe """

    # Private Instance Attributes:
    # - _dataframe: Pandas dataframe being displayed.
    _dataframe: pd.DataFrame

    def __init__(self, dataframe: pd.DataFrame, parent: Qt = None) -> None:
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent: Qt = QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent: Qt = QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role: Qt = Qt.ItemDataRole) -> None:
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
                   ) -> None:
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])

        return None


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['PySide6', 'PySide6.QtCore', 'pandas'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E1136']
    })
