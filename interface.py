"""Module containing all user interface objects, including the initial main window,
the ingest window, the training window, and the predictions window."""
from __future__ import annotations

from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QPushButton


from interface_ingest import IngestWindow
from interface_train import TrainingWindow
from interface_predict import PredictionWindow
from configuration import Config


class MainWindow(QWidget):
    """Primary and initial window which is used as a menu launcher
    for subwindows."""
    # Private Instance Attributes:
    # - _ingest: Ingest subwindow for data ingestion.
    # -_training: Training subwindow for model training.
    # -_predict: Prediction subwindow for model predictions.
    _ingest: IngestWindow
    _training: TrainingWindow
    _predict: PredictionWindow

    def __init__(self, config: Config) -> None:
        super().__init__()

        hbox = QtWidgets.QHBoxLayout(self)

        btn_ingest = QPushButton('Ingest Data', self)
        btn_ingest.resize(btn_ingest.sizeHint())
        btn_ingest.clicked.connect(self.open_ingest)

        btn_train = QPushButton('Train Model', self)
        btn_train.resize(btn_train.sizeHint())
        btn_train.clicked.connect(self.open_training)

        btn_predict = QPushButton('Predict', self)
        btn_predict.resize(btn_predict.sizeHint())
        btn_predict.clicked.connect(self.open_prediction)

        btn_quit = QPushButton('Force Quit', self)
        btn_quit.clicked.connect(QApplication.instance().quit)
        btn_quit.resize(btn_quit.sizeHint())

        hbox.addWidget(btn_ingest)
        hbox.addWidget(btn_train)
        hbox.addWidget(btn_predict)
        hbox.addWidget(btn_quit)

        self.setWindowTitle('Crypto Futures')
        self.show()
        self.setFixedSize(self.size())

        self._ingest = IngestWindow(config)
        self._ingest.setWindowTitle('Ingest Data')

        self._training = TrainingWindow()
        self._training.setWindowTitle('Training')

        self._predict = PredictionWindow()
        self._predict.setWindowTitle('Predictions')

    def open_ingest(self) -> None:
        """Opens the ingest window."""
        self._ingest.show()

    def open_training(self) -> None:
        """Opens the training window."""
        self._training.show()

    def open_prediction(self) -> None:
        """Opens the prediction window."""
        self._predict.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['numpy', 'data_tools', 'configuration',
                          'PySide6', 'PySide6.QtWidgets', 'pandas.core.frame',
                          'interface_train', 'interface_predict',
                          'interface_ingest'],
        'allowed-io': ['__init__', 'run'],
        'max-line-length': 100,
        'disable': ['E1136']
    })
