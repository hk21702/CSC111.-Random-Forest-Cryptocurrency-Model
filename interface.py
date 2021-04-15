"""Module containing all user interface objects, including the initial main window,
the ingest window, the training window, and the predictions window."""
from __future__ import annotations
from datetime import datetime
from typing import Union
import pickle
from numpy import ndarray

import pandas as pd
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6.QtCore import QRunnable, QThreadPool, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (QApplication, QDialog, QTableView, QWidget, QPushButton,
                               QMessageBox, QGridLayout, QDialogButtonBox,
                               QGroupBox, QLabel, QComboBox, QListWidget, QLayout,
                               QLineEdit, QCalendarWidget, QPlainTextEdit, QMainWindow)

import data_ingest
from interface_pandas import PandasModel
from configuration import Config
from data_classes import ForestArgs, WindowArgs
from random_forest import RandomForest, save_model, load_model, load_corresponding_dataframes


class MainWindow(QWidget):
    """Primary and initial window which is used as a menu launcher
    for subwindows."""

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


class TrainingWindow(QDialog):
    """Subwindow dedicated to random forest training functions."""

    txt_var: dict[str, QLineEdit] = {}
    lists: dict[str, Union[QListWidget, QComboBox]] = {}

    _main_layout: QGridLayout
    _button_box: QDialogButtonBox
    _options_group_box: QGroupBox

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

        self.lists['data'] = QListWidget()
        self.lists['data'].setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)

        self.lists['target'] = QComboBox()
        self.lists['f_target'] = QComboBox()

        self.lists['target'].currentTextChanged.connect(
            self.refresh_f_target_list)

        self.refresh_lists()

        dataset_label = QLabel("Datasets:")
        targets_label = QLabel("Targets:")
        f_target_label = QLabel("Target Feature:")

        left_options.addWidget(dataset_label, 0, 0)
        left_options.addWidget(self.lists['data'], 1, 0)

        right_options.addWidget(targets_label, 0, 0)
        right_options.addWidget(self.lists['target'], 1, 0)
        right_options.addWidget(f_target_label, 2, 0)
        right_options.addWidget(self.lists['f_target'], 3, 0)

        right_options.addLayout(self._create_num_options_group_box(), 4, 0)

        name_validator = QRegularExpressionValidator(r'^[\w\-. ]+$')
        name_label = QLabel("Model Name:")
        self.txt_var['model_name'] = QLineEdit()
        self.txt_var['model_name'].setValidator(name_validator)
        right_options.addWidget(name_label, 5, 0)
        right_options.addWidget(self.txt_var['model_name'], 6, 0)

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

        self.txt_var['window_size'] = QLineEdit()
        self.txt_var['target_shift'] = QLineEdit()
        self.txt_var['n_trees'] = QLineEdit('10')
        self.txt_var['sample_sz'] = QLineEdit('300')

        for dataset in self.txt_var.values():
            dataset.setValidator(validator)

        num_options_layout.addWidget(window_size_label, 0, 0)
        num_options_layout.addWidget(self.txt_var['window_size'], 1, 0)

        num_options_layout.addWidget(target_shift_label, 0, 1)
        num_options_layout.addWidget(self.txt_var['target_shift'], 1, 1)

        num_options_layout.addWidget(n_trees_label, 2, 0)
        num_options_layout.addWidget(self.txt_var['n_trees'], 3, 0)

        num_options_layout.addWidget(sample_sz_label, 2, 1)
        num_options_layout.addWidget(self.txt_var['sample_sz'], 3, 1)

        return num_options_layout

    def refresh_lists(self) -> None:
        """Refreshes avaliable datasets for training."""
        self.lists['data'].clear()
        self.lists['target'].clear()

        data_list = data_ingest.get_avaliable_data(search_type='data')

        self.lists['data'].addItems(data_list)
        self.lists['target'].addItems(data_list)

    def refresh_f_target_list(self) -> None:
        """Refreshes avaliable features for the selected target."""
        dataset = self.lists['target'].currentText()
        self.lists['f_target'].clear()
        if dataset != '':
            try:
                features = data_ingest.load_data(dataset).columns
                self.lists['f_target'].addItems(features)
            except:
                self._error_event(f'{dataset} is an invalid dataset.')
                raise

    def _train(self) -> None:
        """Trains a model and saves it based on current params."""
        self.setEnabled(False)

        for x in self.txt_var.items():
            if x[1].text() == '':
                self._error_event(
                    f'{x[0]} is a required input.')
                self.setEnabled(True)
                return None

        dataframes = self.lists['data'].selectedItems()
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
        target_name = self.lists['target'].currentText()
        try:
            target = pd.DataFrame(data_ingest.load_data(
                target_name)[self.lists['f_target'].currentText()])
        except:
            response = self._error_event(
                f'{target_name} is an invalid dataset.', choice=True)
            if response == QtWidgets.QMessageBox.Abort:
                self.setEnabled(True)
                return None
            raise

        req_features = {sym.text().split('_', maxsplit=1)[
            0] for sym in dataframes}

        window_args = WindowArgs(int(self.txt_var['window_size'].text()),
                                 int(self.txt_var['target_shift'].text()),
                                 req_features, dfs, target,
                                 self.lists['f_target'].currentText())
        forest_args = ForestArgs(
            int(self.txt_var['n_trees'].text()), int(self.txt_var['sample_sz'].text()))

        pool = QThreadPool.globalInstance()

        training = TrainingThread(
            window_args, forest_args, self.txt_var['model_name'].text())

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

        print('Model Created')

        save_model(self.name, model)

        print('Model Saved')

        self.signals.float_result.emit(model.accuracy)


class WorkerSignals(QtCore.QObject):
    """ Defines signals avaliable from a worker thread.

        - float_result: float data returned from processing.
    """
    float_result: Signal = Signal(float)


class PredictionWindow(QWidget):
    """Subwindow dedicated to random forest prediction functions."""

    _predict_btn: QDialogButtonBox
    _model: QComboBox
    _model_info: QPlainTextEdit
    _main_layout: QGridLayout
    _button_box: QDialogButtonBox
    _target_date: QCalendarWidget
    _plot_window: QMainWindow

    def __init__(self) -> None:
        super().__init__()

        self._create_button_box()
        main_layout = QGridLayout()
        main_layout.addWidget(self._create_options_group_box(), 0, 0)
        main_layout.addWidget(self._button_box, 1, 0)

        main_layout.setSizeConstraint(QLayout.SetMinimumSize)

        self._main_layout = main_layout
        self.setLayout(self._main_layout)

        self.setWindowTitle('Predict')
        self._plot_window = QMainWindow()

    def _create_button_box(self) -> None:
        """Creates the lower control buttons at the bottom of the window."""
        self._button_box = QDialogButtonBox()

        self._predict_btn = self._button_box.addButton(
            'Predict', QDialogButtonBox.ActionRole)
        refresh_btn = self._button_box.addButton(
            'Refresh &Options', QDialogButtonBox.ActionRole)

        self._predict_btn.clicked.connect(self._predict)
        refresh_btn.clicked.connect(self._refresh_lists)

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

    def _refresh_lists(self) -> None:
        """Refreshes avaliable datasets for training."""
        self._model.clear()

        data_list = data_ingest.get_avaliable_data(search_type='model')

        self._model.addItems(data_list)

    def _refresh_model_info(self) -> None:
        """Refreshes avaliable features for the selected target."""
        self._model_info.clear()

        model_name = self._model.currentText()

        model = self._selected_model
        if model is None:
            self._predict_btn.setEnabled(False)
            return None

        req_features = model.window.req_features

        avaliable_sym = data_ingest.get_avaliable_sym()

        if len(req_features - avaliable_sym) != 0:
            self._error_event(
                f'Missing required data for {model_name}: {req_features - avaliable_sym}')
            self._predict_btn.setEnabled(False)
            return None

        dfs = load_corresponding_dataframes(model)
        grouped_dataframe = data_ingest.create_grouped_dataframe(dfs)

        date_offset = pd.DateOffset(days=model.window.target_shift)

        self._target_date.setMaximumDate(
            grouped_dataframe.index.max() + date_offset)
        self._target_date.setMinimumDate(
            grouped_dataframe.index.min() + date_offset)

        self._display_model_info(model)

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
        except data_ingest.MissingData:
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


class IngestWindow(QDialog):
    """Subwindow dedicated to data ingest functions."""

    txt_var: dict[str, QLineEdit] = {}
    lists: dict[str, Union[QListWidget, QComboBox]] = {}
    config: Config

    _main_layout: QGridLayout
    _button_box: QDialogButtonBox
    _options_group_box: QGroupBox

    def __init__(self, config: Config) -> None:
        super().__init__()

        self._view = QTableView()
        self._view.horizontalHeader().setCascadingSectionResizes(True)
        self._view.horizontalHeader().setStretchLastSection(True)
        self._view.setAlternatingRowColors(True)
        self._view.setSelectionBehavior(QTableView.SelectRows)

        self.config = config

        self._create_options_group_box()
        self._create_button_box()

        main_layout = QGridLayout()
        main_layout.addWidget(self._options_group_box, 0, 0)
        main_layout.addWidget(self._button_box, 1, 0)
        main_layout.addWidget(self._view, 2, 0)
        main_layout.setSizeConstraint(QLayout.SetMinimumSize)

        self._main_layout = main_layout
        self.setLayout(self._main_layout)

        self.setWindowTitle('Train Model')

    def _create_button_box(self) -> None:
        """Creates the lower control buttons at the bottom of the window."""
        self._button_box = QDialogButtonBox()

        ingest_btn = self._button_box.addButton(
            'Get Data', QDialogButtonBox.ActionRole)
        delete_btn = self._button_box.addButton(
            'Delete Dataset', QDialogButtonBox.ActionRole)
        refresh_btn = self._button_box.addButton(
            'Refresh &Options', QDialogButtonBox.ActionRole)

        ingest_btn.clicked.connect(self._ingest)
        delete_btn.clicked.connect(self._delete)
        refresh_btn.clicked.connect(self._refresh_lists)

    def _create_options_group_box(self) -> None:
        """Creates the group of training options."""
        self._options_group_box = QGroupBox("Options")

        options_layout = QGridLayout()
        left_options = QGridLayout()
        right_options = QGridLayout()

        self.lists['data'] = QListWidget()
        self.lists['data'].setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.lists['data'].currentTextChanged.connect(self._refresh_pandas)

        self.lists['type'] = QComboBox()
        self.lists['type'].addItem('Crypto Currencies Daily')
        self.lists['type'].addItem('Standard Time Series Daily Adjusted')

        self._refresh_lists()

        validator = QRegularExpressionValidator(r'^[\w\-. ]+$')

        dataset_label = QLabel("Avaliable Datasets:")
        search_type_label = QLabel("Symbol/Search Type:")
        search_label = QLabel("Symbol/Search Term:")
        name_label = QLabel("Dataset Name:")

        left_options.addWidget(dataset_label, 0, 0)
        left_options.addWidget(self.lists['data'], 1, 0)

        right_options.addWidget(search_type_label, 0, 0)
        right_options.addWidget(self.lists['type'], 1, 0)

        self.txt_var['ds_name'] = QLineEdit()
        self.txt_var['data_search'] = QLineEdit()
        self.txt_var['ds_name'].setValidator(validator)
        self.txt_var['data_search'].setValidator(validator)

        right_options.addWidget(search_label, 2, 0)
        right_options.addWidget(self.txt_var['data_search'], 3, 0)

        right_options.addWidget(name_label, 5, 0)
        right_options.addWidget(self.txt_var['ds_name'], 6, 0)

        options_layout.addLayout(left_options, 0, 0)
        options_layout.addLayout(right_options, 0, 1)

        options_layout.setColumnStretch(0, 1)

        self._options_group_box.setLayout(options_layout)

    def _refresh_lists(self) -> None:
        """Refreshes avaliable datasets for training."""
        self.lists['data'].clear()

        data_list = data_ingest.get_avaliable_data(search_type='data')

        self.lists['data'].addItems(data_list)

    def _refresh_pandas(self) -> None:
        """Refreshes the pandas table viewer."""
        df = self.lists['data'].currentItem().text()
        if df in data_ingest.get_avaliable_data(search_type='data'):
            df = data_ingest.load_data(df)
            model = PandasModel(df)
            self._view.setModel(model)

    def _ingest(self) -> None:
        self.setEnabled(False)
        search_type = self.lists['type'].currentText()
        search_term = self.txt_var['data_search'].text()
        if search_type == 'Crypto Currencies Daily':
            try:
                data = data_ingest.async_get([data_ingest.get_cc_daily(
                    search_term, self.config, cache=False)])
                type = data_ingest.AVDataTypes.CryptoCurrenciesDaily
            except data_ingest.RateLimited:
                self._error_event(
                    'You are being rate limited. Check Alpha Vantage API key or wait.')
                self.setEnabled(True)
                return None
            except data_ingest.UnknownAVType:
                self._error_event(
                    f'{search_term} is an invalid symbol for {search_type}')
                self.setEnabled(True)
                return None
        elif search_type == 'Standard Time Series Daily Adjusted':
            try:
                data = data_ingest.async_get([data_ingest.get_ts_daily_adjusted(
                    search_term, self.config, cache=False)])
                type = data_ingest.AVDataTypes.TimeSeriesDailyAdjusted
            except data_ingest.RateLimited:
                self._error_event(
                    'You are being rate limited. Check Alpha Vantage API key or wait.')
                self.setEnabled(True)
                return None

            except data_ingest.UnknownAVType:
                self._error_event(
                    f'{search_term} is an invalid symbol for {search_type}')
                self.setEnabled(True)
                return None
        else:
            self._error_event('Invalid search type.')
            return None
        name = self.txt_var['ds_name'].text()
        if name == '':
            name = data_ingest.name_generator(
                search_term, type)
        else:
            name = data_ingest.name_generator(
                search_term, type, name)
        current_datasets = data_ingest.get_avaliable_data(search_type='data')
        if name in current_datasets:
            response = self._error_event(
                f'{name} will be overwritten.', choice=True)
            if response == QMessageBox.Abort:
                self.setEnabled(True)
                return None

        data_ingest.save_data(name, data[0])

        self._refresh_lists()
        self.setEnabled(True)
        return None

    def _delete(self) -> None:
        """Deletes the currently selected dataset."""
        self.setEnabled(False)
        name = self.lists['data'].selectedItems()[0].text()

        response = self._error_event(
            f'Are you sure you want to delete {name}?',
            choice=True, btn=QMessageBox.Ok)
        if response == QMessageBox.Abort:
            self.setEnabled(True)
            return None

        data_ingest.delete_data(name)
        self._refresh_lists()
        self.setEnabled(True)
        return None

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
        'extra-imports': ['numpy', 'pandas', 'data_tools',
                          'data_ingest', 'data_classes', 'PySide6',
                          'PySide6.QtCore', 'PySide6.QtGui', 'contextlib',
                          'PySide6.QtWidgets', 'pandas.core.frame',
                          'random_forest', 'pickle', 'pyqtgraph', 'datetime'],
        'allowed-io': ['__init__', 'run'],
        'max-line-length': 100,
        'disable': ['E1136']
    })
