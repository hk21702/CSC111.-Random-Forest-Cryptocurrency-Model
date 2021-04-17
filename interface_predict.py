"""Module containing the PredictionWindow class, the subwindow used for any user inputs
to make a prediction using a random forest model."""
from __future__ import annotations
from datetime import datetime
from typing import Union
import pickle
from numpy import ndarray

import pandas as pd
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from PySide6 import QtWidgets
from PySide6.QtWidgets import (QWidget, QMessageBox, QGridLayout, QDialogButtonBox,
                               QGroupBox, QLabel, QComboBox, QLayout,
                               QCalendarWidget, QPlainTextEdit, QMainWindow)

import data_ingest
import common_exceptions as ce
from random_forest import RandomForest, load_model, load_corresponding_dataframes


class PredictionWindow(QWidget):
    """Subwindow dedicated to random forest prediction functions."""

    # Private Instance Attributes:
    # - _predict_btn: Button that signals for a prediction to be made.
    # - _predict_btn: Button that signals for the selected model to be deleted.
    # - _model: The RandomForest model currently loaded.
    # - _model_info: QPlainTextEdit widget displaying information about the selected model.
    # - _target_date: Calendar widget for the user to select a target prediction date.
    # -_plot_window: Window for displaying historical information with a prediction.
    _predict_btn: QDialogButtonBox
    _delete_btn: QDialogButtonBox
    _model: QComboBox
    _model_info: QPlainTextEdit
    _target_date: QCalendarWidget
    _plot_window: QMainWindow

    def __init__(self) -> None:
        super().__init__()

        main_layout = QGridLayout()
        button_box = self._create_button_box()
        main_layout.addWidget(self._create_options_group_box(), 0, 0)
        main_layout.addWidget(button_box, 1, 0)

        main_layout.setSizeConstraint(QLayout.SetMinimumSize)

        self.setLayout(main_layout)

        self._refresh_model_info()
        self.setWindowTitle('Predict')
        self._plot_window = QMainWindow()

    def show(self) -> None:
        """Override of QWidget's show() function.

        Refreshes window and then shows the window.
        """
        self._refresh_lists()
        return super().show()

    @property
    def _selected_model(self) -> Union[RandomForest, None]:
        """Gets the currently selected model."""
        model_name = self._model.currentText()
        if model_name != '':
            try:
                return load_model(model_name)
            except pickle.UnpicklingError:
                self._error_event(f'{model_name} is an invalid model.')
                return None
        else:
            return None

    def _create_button_box(self) -> QDialogButtonBox:
        """Creates the lower control buttons at the bottom of the window."""
        button_box = QDialogButtonBox()

        self._predict_btn = button_box.addButton(
            'Predict', QDialogButtonBox.ActionRole)
        self._delete_btn = button_box.addButton(
            'Delete Model', QDialogButtonBox.ActionRole)
        refresh_btn = button_box.addButton(
            'Refresh &Options', QDialogButtonBox.ActionRole)

        self._predict_btn.clicked.connect(self._predict)
        refresh_btn.clicked.connect(self._refresh_lists)
        self._delete_btn.clicked.connect(self._delete)

        return button_box

    def _create_options_group_box(self) -> QGroupBox:
        """Returns the group of prediction options."""
        options_group_box = QGroupBox("Options")

        options_layout = QGridLayout()
        left_options = QGridLayout()
        right_options = QGridLayout()

        date_label = QLabel("Target Date:")
        self._target_date = QCalendarWidget()

        left_options.addWidget(date_label, 0, 0)
        left_options.addWidget(self._target_date, 1, 0, 1, 3)

        left_options.setColumnStretch(0, 1)

        self._model = QComboBox()
        self._model_info = QPlainTextEdit()
        self._model_info.setReadOnly(True)

        self._model.currentTextChanged.connect(
            self._refresh_model_info)

        self._refresh_lists()

        models_label = QLabel("Models:")
        info_label = QLabel("Model Information:")

        right_options.addWidget(models_label, 0, 0)
        right_options.addWidget(self._model, 1, 0)

        right_options.addWidget(info_label, 2, 0)
        right_options.addWidget(self._model_info, 3, 0)

        options_layout.addLayout(left_options, 0, 0)
        options_layout.addLayout(right_options, 0, 1)

        options_group_box.setLayout(options_layout)

        return options_group_box

    def _delete(self) -> None:
        """Deletes the currently selected dataset."""
        self.setEnabled(False)
        name = self._model.currentText()

        warning = f'Are you sure you want to delete {name}?'

        response = QMessageBox.warning(self, self.tr("Delete Model"),
                                       warning, QMessageBox.Yes,
                                       QMessageBox.No)

        if response == QMessageBox.Yes:
            data_ingest.delete_data(name, file_type='model')
            self._refresh_lists()

        self.setEnabled(True)

    def _refresh_lists(self) -> None:
        """Refreshes avaliable datasets for training."""
        self._model.clear()

        data_list = data_ingest.get_avaliable_data(search_type='model')

        self._model.addItems(data_list)

    def _refresh_model_info(self) -> None:
        """Refreshes avaliable features for the selected target."""
        self._predict_btn.setEnabled(False)
        self._target_date.setEnabled(False)
        self._model_info.clear()

        model_name = self._model.currentText()

        model = self._selected_model
        if model is None:
            self._delete_btn.setEnabled(False)
            return None

        self._delete_btn.setEnabled(True)

        self._display_model_info(model)

        req_features = model.window.req_features

        avaliable_sym = data_ingest.get_avaliable_sym()

        if len(req_features - avaliable_sym) != 0:
            self._error_event(
                f'Missing required data for {model_name}: {req_features - avaliable_sym}')
            return None

        dfs = load_corresponding_dataframes(model)
        grouped_dataframe = data_ingest.create_grouped_dataframe(dfs)

        date_offset = pd.DateOffset(days=model.window.target_shift)

        self._target_date.setMaximumDate(
            grouped_dataframe.index.max() + date_offset)
        self._target_date.setMinimumDate(
            grouped_dataframe.index.min() + date_offset)

        self._target_date.setEnabled(True)
        self._predict_btn.setEnabled(True)

        return None

    def _display_model_info(self, model: RandomForest) -> None:
        """Updates model info box to display current model's information."""
        self._model_info.appendPlainText(
            f'Target Feature Name: \n{model.window.target_lbl}')
        self._model_info.appendPlainText('Window Information:')
        self._model_info.appendPlainText(
            f'\t- Window Size: {model.window.window_size}')
        self._model_info.appendPlainText(
            f'\t- Target Shift: {model.window.target_shift}')
        self._model_info.appendPlainText(
            f'\t- Required Features: {model.window.req_features}')
        self._model_info.appendPlainText(
            'Forest Information:')
        self._model_info.appendPlainText(
            f'\t- Number of Trees: {model.forest.n_trees}')
        self._model_info.appendPlainText(
            f'\t- Tree Max Depth: {model.forest.max_depth}')
        self._model_info.appendPlainText(
            f'\t- Seed: {model.forest.seed}')

    def _predict(self) -> None:
        """Creates a model prediction using the selected target date and model."""
        self.setEnabled(False)
        self._predict_btn.setEnabled(False)
        model = self._selected_model
        if model is None:
            self.setEnabled(True)
            return None

        target_date = self._target_date.selectedDate().toPython()

        try:
            dfs = load_corresponding_dataframes(model)
            prediction_input = data_ingest.create_input(model.window.window_size,
                                                        model.window.target_shift,
                                                        target_date,
                                                        dfs)
        except ce.MissingData:
            self._error_event(
                'Missing required data. Could be that loaded datasets have holes.')
            self.setEnabled(True)
            return None

        prediction = model.predict(prediction_input)

        historical_dfs = load_corresponding_dataframes(model, 'target')
        if len(historical_dfs) == 0:
            self._prediction_historical_error(prediction)
        else:
            self._plot_prediction(historical_dfs, model,
                                  prediction, target_date)

        self._predict_btn.setEnabled(True)
        self.setEnabled(True)
        return None

    def _plot_prediction(self, historical_dfs: list[pd.DataFrame], model: RandomForest,
                         prediction: ndarray, target_date: datetime.date) -> None:
        """Opens a window with a plot of the historical target data as well as the prediction
        the model made."""
        hdf = historical_dfs[0]
        for frame in historical_dfs[1:]:
            hdf = hdf.combine_first(frame)
        window_end = target_date - \
            pd.DateOffset(days=model.window.target_shift)
        window_start = window_end - pd.DateOffset(days=30 - 1)
        hdf = pd.Series(
            hdf.loc[window_end: window_start][model.window.target_lbl])

        hdf_data = hdf.to_list()
        hdf_dates = hdf.index

        hdf_dates = [ts.to_pydatetime().timestamp() for ts in hdf_dates]

        b_axis = pg.DateAxisItem(orientation='bottom')
        b_axis.setLabel('Date')
        plot = PlotWidget(
            axisItems={'bottom': b_axis})

        target_time = datetime.combine(target_date, datetime.min.time())

        plot.addLegend()
        plot.plot(x=hdf_dates, y=hdf_data,
                  name=f'Historical {model.window.target_lbl}')
        plot.plot(x=[target_time.timestamp()],
                  y=prediction, pen=None, symbol='o',
                  name=f'Predicted Value: {prediction[0]}')
        model_name = self._model.currentText()
        self._plot_window.setWindowTitle(
            f'{model_name} Prediction')
        self._plot_window.setCentralWidget(plot)
        self._plot_window.show()

    def _prediction_historical_error(self, prediction: list) -> None:
        """Displays a message for when historical target is unavalable such that
        a graph can't be made."""
        QMessageBox.information(self, self.tr("Information"),
                                f'Prediction was: {prediction}. \n '
                                'Unable to display graph due to missing historical data.',
                                QtWidgets.QMessageBox.Ok)

    def _error_event(self, error: str,
                     choice: bool = False,
                     btn: QMessageBox = QMessageBox.Abort) -> Union[QMessageBox.Ignore,
                                                                    QMessageBox.Abort,
                                                                    None]:
        """Displays an error message with the given error."""
        if choice:
            response = QMessageBox.critical(self, self.tr("Error"),
                                            error, btn,
                                            QMessageBox.Ignore)
            return response
        else:
            QMessageBox.critical(self, self.tr("Error"),
                                 error, QMessageBox.Ok)
            return None


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['__future__', 'datetime', 'common_exceptions',
                          'typing', 'pickle', 'numpy', 'pandas', 'pyqtgraph',
                          'PySide6', 'PySide6.QtWidgets', 'data_ingest', 'random_forest'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E1136']
    })
