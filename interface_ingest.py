"""Module containing the IngestWindow class, the subwindow used for any user inputs
to get and save datasets from any source."""
from __future__ import annotations
from typing import Union
from datetime import datetime, timedelta

import pandas as pd
from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (QDialog, QTableView, QMessageBox, QGridLayout, QDialogButtonBox,
                               QGroupBox, QLabel, QComboBox, QListWidget, QLayout,
                               QLineEdit)
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq

import data_ingest
import google_trends
import common_exceptions as ce
from data_ingest import IngestTypes
from interface_pandas import PandasModel
from configuration import Config

GOOGLE_TRENDS_SHIFT = timedelta(days=1500)


class IngestWindow(QDialog):
    """Subwindow dedicated to data ingest functions.

    Instance attributes:
        - config: Config object holding configuration information
        including API keys.
    """
    config: Config

    # Private Instance Attributes:
    # - _main_layout: Main grid layout
    # - _button_box: Button box holding main control buttons.
    # - _options_group_box: Group box holding most UI widgets
    # - _view: QTableView holding the pandas table viewer.
    # - _pool: QThreadPool for handeling multithreaded functions.
    # - _txt_var: Dictionary holding line edit widgets.
    # - _lists: Dictionary holding the dataset list widget and search type combo widget.
    _main_layout: QGridLayout
    _button_box: QDialogButtonBox
    _options_group_box: QGroupBox
    _view: QTableView
    _pool: QThreadPool
    _delete_btn: QDialogButtonBox
    _txt_var: dict[str, QLineEdit] = {}
    _lists: dict[str, Union[QListWidget, QComboBox]] = {}

    def __init__(self, config: Config) -> None:
        super().__init__()

        self._pool = QThreadPool.globalInstance()

        self._view = QTableView()
        self._view.horizontalHeader().setCascadingSectionResizes(True)
        self._view.horizontalHeader().setStretchLastSection(True)
        self._view.setAlternatingRowColors(True)
        self._view.setSelectionBehavior(QTableView.SelectRows)

        self.config = config

        self._init_options_group_box()
        self._init_button_box()

        main_layout = QGridLayout()
        main_layout.addWidget(self._options_group_box, 0, 0)
        main_layout.addWidget(self._button_box, 1, 0)
        main_layout.addWidget(self._view, 2, 0)
        main_layout.setSizeConstraint(QLayout.SetMinimumSize)

        self._main_layout = main_layout
        self.setLayout(self._main_layout)
        self._type_changed()

        self.setWindowTitle('Train Model')

    def show(self) -> None:
        """Override of QWidget's show() function.

        Refreshes window and then shows the window.
        """
        super().show()
        self._refresh_lists()
        self._refresh_pandas()

    def _init_button_box(self) -> None:
        """Creates the lower control buttons at the bottom of the window."""
        self._button_box = QDialogButtonBox()

        ingest_btn = self._button_box.addButton(
            'Get Data', QDialogButtonBox.ActionRole)
        self._delete_btn = self._button_box.addButton(
            'Delete Dataset', QDialogButtonBox.ActionRole)
        refresh_btn = self._button_box.addButton(
            'Refresh &Options', QDialogButtonBox.ActionRole)

        ingest_btn.clicked.connect(self._ingest)
        self._delete_btn.clicked.connect(self._delete)
        refresh_btn.clicked.connect(self._refresh_lists)

    def _init_options_group_box(self) -> None:
        """Creates the group of training options."""
        self._options_group_box = QGroupBox("Options")

        options_layout = QGridLayout()
        left_options = QGridLayout()
        right_options = QGridLayout()

        self._lists['data'] = QListWidget()
        self._lists['data'].setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self._lists['data'].currentTextChanged.connect(self._refresh_pandas)

        self._lists['type'] = QComboBox()
        for dt in IngestTypes:
            self._lists['type'].addItem(dt.value)

        self._refresh_lists()

        self._lists['type'].currentTextChanged.connect(self._type_changed)

        validator = QRegularExpressionValidator(r'^[\w\-. ]+$')
        cat_validator = QRegularExpressionValidator(r'^[0-9]\d*$')

        dataset_label = QLabel("Avaliable Datasets:")
        search_type_label = QLabel("Symbol/Search Type:")
        search_label = QLabel("Symbol/Search Term:")
        name_label = QLabel("Dataset Name:")
        cat_label = QLabel("Trends Category Code:")

        left_options.addWidget(dataset_label, 0, 0)
        left_options.addWidget(self._lists['data'], 1, 0)

        right_options.addWidget(search_type_label, 0, 0)
        right_options.addWidget(self._lists['type'], 1, 0)

        self._txt_var['ds_name'] = QLineEdit()
        self._txt_var['data_search'] = QLineEdit()
        self._txt_var['search_cat'] = QLineEdit()
        self._txt_var['ds_name'].setValidator(validator)
        self._txt_var['data_search'].setValidator(validator)
        self._txt_var['search_cat'].setValidator(cat_validator)

        self._txt_var['search_cat'].setPlaceholderText('0')

        right_options.addWidget(search_label, 2, 0)
        right_options.addWidget(self._txt_var['data_search'], 3, 0)

        right_options.addWidget(name_label, 4, 0)
        right_options.addWidget(self._txt_var['ds_name'], 5, 0)

        right_options.addWidget(cat_label, 6, 0)
        right_options.addWidget(self._txt_var['search_cat'], 7, 0)

        options_layout.addLayout(left_options, 0, 0)
        options_layout.addLayout(right_options, 0, 1)

        options_layout.setColumnStretch(0, 1)

        self._options_group_box.setLayout(options_layout)

    def _refresh_lists(self) -> None:
        """Refreshes avaliable datasets for training."""
        self._lists['data'].clear()

        data_list = data_ingest.get_avaliable_data(search_type='data')

        self._lists['data'].addItems(data_list)

    def _refresh_pandas(self) -> None:
        """Refreshes the pandas table viewer."""
        df = self._lists['data'].currentItem()
        if df is not None:
            df = df.text()
            if df in data_ingest.get_avaliable_data(search_type='data'):
                df = data_ingest.load_data(df)
                model = PandasModel(df)
                self._view.setModel(model)
                self._delete_btn.setEnabled(True)
                return

        self._delete_btn.setEnabled(False)

    def _type_changed(self) -> None:
        """Updates widget based on type combo box change."""
        if self._lists['type'].currentText() == 'Google Trends':
            self._txt_var['ds_name'].setEnabled(False)
            self._txt_var['search_cat'].setEnabled(True)
        else:
            self._txt_var['ds_name'].setEnabled(True)
            self._txt_var['search_cat'].setEnabled(False)

    def _ingest(self) -> None:
        """Ingests the user requested data and saves it accordingly."""
        self.setEnabled(False)
        search_type = self._lists['type'].currentText()
        search_type = IngestTypes.from_str(search_type)

        search_term = self._txt_var['data_search'].text().strip()
        self._txt_var['data_search'].setText(search_term)

        if search_type in {IngestTypes.CryptoCurrenciesDaily, IngestTypes.TimeSeriesDailyAdjusted}:
            try:
                if search_type == IngestTypes.CryptoCurrenciesDaily:
                    data = data_ingest.async_get([data_ingest.get_cc_daily(
                        search_term, self.config, cache=False)])[0]
                elif search_type == IngestTypes.TimeSeriesDailyAdjusted:
                    data = data_ingest.async_get([data_ingest.get_ts_daily_adjusted(
                        search_term, self.config, cache=False)])[0]
                else:
                    data = None  # Satisfy PyTa even though raise occurs
                    raise NotImplementedError
            except ce.RateLimited:
                self._error_event(
                    'You are being rate limited. Check Alpha Vantage API key or wait.')
                return
            except ce.UnknownAVType:
                self._error_event(
                    f'{search_term} is an invalid symbol for {search_type}')
                return
        elif search_type == IngestTypes.GoogleTrends:
            trends_thread = TrendsThread(
                search_term, self._txt_var['search_cat'].text())

            self._pool.start(trends_thread)

            trends_thread.signals.error.connect(self._error_event)

            trends_thread.signals.dataframe_result.connect(self._ingest_save)
            return
        else:
            self._error_event('Invalid search type.')
            return

        self._ingest_save(data)

    def _ingest_save(self, data: pd.DataFrame) -> None:
        """Generates a name and saves the given dataframe.
        Returns None when complete or on error."""
        search_type = self._lists['type'].currentText()
        search_term = self._txt_var['data_search'].text()

        search_type = IngestTypes.from_str(search_type)
        if search_type == IngestTypes.GoogleTrends:
            cat = self._txt_var['search_cat'].text()

            if cat == '':
                cat = '0'
            name = f'{search_term} {cat};{data_ingest.IngestTypes.GoogleTrends.name}'
        else:
            name = self._txt_var['ds_name'].text()
            if name == '':
                name = data_ingest.name_generator(
                    search_term, search_type)
            else:
                name = data_ingest.name_generator(
                    search_term, search_type, name)

        current_datasets = data_ingest.get_avaliable_data(
            search_type='data')

        if name in current_datasets:
            response = self._error_event(
                f'{name} will be overwritten.', choice=True)
            if response == QMessageBox.Abort:
                self.setEnabled(True)
                return

        data_ingest.save_data(name, data)

        self._refresh_lists()
        self.setEnabled(True)

    def _delete(self) -> None:
        """Deletes the currently selected dataset."""
        self.setEnabled(False)
        name = self._lists['data'].selectedItems()

        if len(name) != 0:
            name = name[0].text()

            warning = f'Are you sure you want to delete {name}?'

            response = QMessageBox.warning(self, self.tr("Delete Dataset"),
                                           warning, QMessageBox.Yes,
                                           QMessageBox.No)
            if response == QMessageBox.Yes:
                data_ingest.delete_data(name, file_type='data')
                self._refresh_lists()

        self.setEnabled(True)

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
            self.setEnabled(True)
            return None


class TrendsThread(QRunnable):
    """Runnable thread that handles the heavy lifting of doing a google search trend query.

        Instance attributes:
        - search_term: the search term for the query
        - cat: google search trends category ID
        - signals: signal to communicate with main thread once training completes.

"""
    search_term: str
    cat: str
    signals: WorkerSignals

    def __init__(self, search_term: str, cat: str) -> None:
        super().__init__(self)
        self.search_term = search_term
        self.cat = cat
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        """Creates a dataframe from a google search trends."""
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - GOOGLE_TRENDS_SHIFT
        trend_req = TrendReq()

        try:
            if self.cat == '':
                data = google_trends.get_daily_trend(
                    trend_req, self.search_term, start_date, end_date)
            else:
                data = google_trends.get_daily_trend(
                    trend_req, self.search_term, start_date, end_date, int(self.cat))
            self.signals.dataframe_result.emit(data)
        except ce.RateLimited:
            self.signals.error.emit("Google Trends aborted. Rate Limited.")
        except ResponseError as e:
            self.signals.error.emit(
                f"Unknown Google Trends Error. Possibly bad category: {e}")


class WorkerSignals(QtCore.QObject):
    """ Defines signals avaliable from a worker thread.

        - dataframe_result: pandas dataframe data returned from processing.
        - error: error information
    """
    dataframe_result: Signal = Signal(pd.DataFrame)
    error: Signal = Signal(str)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['__future__', 'typing', 'PySide6', 'PySide6.QtGui', 'datetime',
                          'PySide6.QtWidgets', 'data_ingest', 'interface_pandas',
                          'pytrends.request', 'PySide6.QtCore', 'google_trends',
                          'configuration', 'pandas', 'pytrends.exceptions',
                          'common_exceptions'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E1136']
    })
