"""Module containing the TrainingWindow class, the subwindow used
for any user inputs involving the creation and training of a random
forest model, as well as additional helper
classes.
"""
from __future__ import annotations
from typing import Union

import pandas as pd
from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6.QtCore import QRunnable, QThreadPool, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (QDialog, QMessageBox, QGridLayout, QDialogButtonBox,
                               QGroupBox, QLabel, QComboBox, QListWidget, QLayout,
                               QLineEdit)

import data_ingest
from data_classes import ForestArgs, WindowArgs
from random_forest import RandomForest, save_model


class TrainingWindow(QDialog):
    """Subwindow dedicated to random forest training functions."""

    # Private Instance Attributes:
    # - _main_layout: Main grid layout
    # - _button_box: Button box holding main control buttons.
    # - _options_group_box: Group box holding most UI widgets
    # - _txt_var: Dictionary holding line edit widgets.
    # - _lists: Dictionary holding the dataset list widget and combo widget
    # for data selection.
    _main_layout: QGridLayout
    _button_box: QDialogButtonBox
    _options_group_box: QGroupBox
    _txt_var: dict[str, QLineEdit] = {}
    _lists: dict[str, Union[QListWidget, QComboBox]] = {}

    def __init__(self) -> None:
        super().__init__()

        self._create_options_group_box()
        self._create_button_box()

        main_layout = QGridLayout()
        main_layout.addWidget(self._options_group_box, 0, 0)
        main_layout.addWidget(self._button_box, 1, 0)
        main_layout.setSizeConstraint(QLayout.SetMinimumSize)

        self._main_layout = main_layout
        self.setLayout(self._main_layout)

        self.setWindowTitle('Train Model')

    def _create_button_box(self) -> None:
        """Creates the lower control buttons at the bottom of the window."""
        self._button_box = QDialogButtonBox()

        train_btn = self._button_box.addButton(
            'Train &Model', QDialogButtonBox.ActionRole)
        refresh_btn = self._button_box.addButton(
            'Refresh &Options', QDialogButtonBox.ActionRole)

        train_btn.clicked.connect(self._train)
        refresh_btn.clicked.connect(self.refresh_lists)

    def _create_options_group_box(self) -> None:
        """Creates the group of training options."""
        self._options_group_box = QGroupBox("Options")

        options_layout = QGridLayout()
        left_options = QGridLayout()
        right_options = QGridLayout()

        self._lists['data'] = QListWidget()
        self._lists['data'].setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)

        self._lists['target'] = QComboBox()
        self._lists['f_target'] = QComboBox()

        self._lists['target'].currentTextChanged.connect(
            self.refresh_f_target_list)

        self.refresh_lists()

        dataset_label = QLabel("Datasets:")
        targets_label = QLabel("Targets:")
        f_target_label = QLabel("Target Feature:")

        left_options.addWidget(dataset_label, 0, 0)
        left_options.addWidget(self._lists['data'], 1, 0)

        right_options.addWidget(targets_label, 0, 0)
        right_options.addWidget(self._lists['target'], 1, 0)
        right_options.addWidget(f_target_label, 2, 0)
        right_options.addWidget(self._lists['f_target'], 3, 0)

        right_options.addLayout(self._create_num_options_group_box(), 4, 0)

        name_validator = QRegularExpressionValidator(r'^[\w\-. ]+$')
        name_label = QLabel("Model Name:")
        self._txt_var['model_name'] = QLineEdit()
        self._txt_var['model_name'].setValidator(name_validator)
        right_options.addWidget(name_label, 5, 0)
        right_options.addWidget(self._txt_var['model_name'], 6, 0)

        options_layout.addLayout(left_options, 0, 0)
        options_layout.addLayout(right_options, 0, 1)

        options_layout.setColumnStretch(0, 1)

        self._options_group_box.setLayout(options_layout)

    def _create_num_options_group_box(self) -> QGridLayout:
        """Returns input layout for inputs involving numbers."""
        num_options_layout = QGridLayout()
        validator = QRegularExpressionValidator(r'^[1-9]\d*$')

        window_size_label = QLabel("Window Size:")
        target_shift_label = QLabel("Target Shift:")
        n_trees_label = QLabel("Trees in Forest:")
        sample_sz_label = QLabel("Sample Size:")

        self._txt_var['window_size'] = QLineEdit()
        self._txt_var['target_shift'] = QLineEdit()
        self._txt_var['n_trees'] = QLineEdit('10')
        self._txt_var['sample_sz'] = QLineEdit('300')

        for dataset in self._txt_var.values():
            dataset.setValidator(validator)

        num_options_layout.addWidget(window_size_label, 0, 0)
        num_options_layout.addWidget(self._txt_var['window_size'], 1, 0)

        num_options_layout.addWidget(target_shift_label, 0, 1)
        num_options_layout.addWidget(self._txt_var['target_shift'], 1, 1)

        num_options_layout.addWidget(n_trees_label, 2, 0)
        num_options_layout.addWidget(self._txt_var['n_trees'], 3, 0)

        num_options_layout.addWidget(sample_sz_label, 2, 1)
        num_options_layout.addWidget(self._txt_var['sample_sz'], 3, 1)

        return num_options_layout

    def refresh_lists(self) -> None:
        """Refreshes avaliable datasets for training."""
        self._lists['data'].clear()
        self._lists['target'].clear()

        data_list = data_ingest.get_avaliable_data(search_type='data')

        self._lists['data'].addItems(data_list)
        self._lists['target'].addItems(data_list)

    def refresh_f_target_list(self) -> None:
        """Refreshes avaliable features for the selected target."""
        dataset = self._lists['target'].currentText()
        self._lists['f_target'].clear()
        if dataset != '':
            try:
                features = data_ingest.load_data(dataset).columns
                self._lists['f_target'].addItems(features)
            except:
                self._error_event(f'{dataset} is an invalid dataset.')
                raise

    def _train(self) -> None:
        """Trains a model and saves it based on current params."""
        self.setEnabled(False)

        for x in self._txt_var.items():
            if x[1].text() == '':
                self._error_event(
                    f'{x[0]} is a required input.')
                self.setEnabled(True)
                return None

        dataframes = self._lists['data'].selectedItems()
        dfs = []
        for x in dataframes:
            try:
                dfs.append(data_ingest.load_data(x.text()))
            except:
                response = self._error_event(
                    f'{x.text()} is an invalid dataset.', choice=True)
                if response == QtWidgets.QMessageBox.Abort:
                    self.setEnabled(True)
                    return None
                raise
        target_name = self._lists['target'].currentText()
        try:
            target = pd.DataFrame(data_ingest.load_data(
                target_name)[self._lists['f_target'].currentText()])
        except:
            response = self._error_event(
                f'{target_name} is an invalid dataset.', choice=True)
            if response == QtWidgets.QMessageBox.Abort:
                self.setEnabled(True)
                return None
            raise

        req_features = {sym.text().split('_', maxsplit=1)[
            0] for sym in dataframes}

        window_args = WindowArgs(int(self._txt_var['window_size'].text()),
                                 int(self._txt_var['target_shift'].text()),
                                 req_features, dfs, target,
                                 self._lists['f_target'].currentText())
        forest_args = ForestArgs(
            int(self._txt_var['n_trees'].text()), int(self._txt_var['sample_sz'].text()))

        pool = QThreadPool.globalInstance()

        training = TrainingThread(
            window_args, forest_args, self._txt_var['model_name'].text())

        pool.start(training)

        training.signals.float_result.connect(self._training_complete_event)

        return None

    def _training_complete_event(self, return_value: float) -> None:
        """Display information about completed model training."""
        QMessageBox.information(self, self.tr("Information"),
                                f'Model created and saved with final R-Squared of: {return_value}',
                                QtWidgets.QMessageBox.Ok)
        self.setEnabled(True)

    def _error_event(self, error: str,
                     choice: bool = False) -> Union[QtWidgets.QMessageBox.Ignore,
                                                    QtWidgets.QMessageBox.Abort,
                                                    None]:
        """Displays an error message with the given error."""
        if choice:
            response = QMessageBox.critical(self, self.tr("Error"),
                                            error, QtWidgets.QMessageBox.Abort,
                                            QtWidgets.QMessageBox.Ignore)
            return response
        else:
            QMessageBox.critical(self, self.tr("Error"),
                                 error, QtWidgets.QMessageBox.Ok)
            return None


class TrainingThread(QRunnable):
    """Runnable thread that handles the heavy lifting of creating, training and saving models."""
    window_args: WindowArgs
    forest_args: ForestArgs
    name: str
    signals: WorkerSignals

    def __init__(self, window_args: WindowArgs, forest_args: ForestArgs, name: str) -> None:
        super().__init__(self)
        self.window_args = window_args
        self.forest_args = forest_args
        self.name = name
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Creates a random forest model, trains it, and saves it."""

        model = RandomForest(self.window_args, self.forest_args)

        save_model(self.name, model)

        self.signals.float_result.emit(model.accuracy)


class WorkerSignals(QtCore.QObject):
    """ Defines signals avaliable from a worker thread.

        - float_result: float data returned from processing.
    """
    float_result: Signal = Signal(float)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['__future__', 'typing', 'PySide6', 'PySide6.QtGui',
                          'PySide6.QtWidgets', 'data_ingest', 'interface_pandas',
                          'random_forest', 'pandas', 'PySide6.QtCore', 'data_classes'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E1136']
    })
