"""Module containing the IngestWindow class, the subwindow used for any user inputs
to get and save datasets from any source."""
from __future__ import annotations
from typing import Union

from PySide6 import QtWidgets
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (QDialog, QTableView, QMessageBox, QGridLayout, QDialogButtonBox,
                               QGroupBox, QLabel, QComboBox, QListWidget, QLayout,
                               QLineEdit)

import data_ingest
from interface_pandas import PandasModel
from configuration import Config


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
    # - _txt_var: Dictionary holding line edit widgets.
    # - _lists: Dictionary holding the dataset list widget and search type combo widget.
    _main_layout: QGridLayout
    _button_box: QDialogButtonBox
    _options_group_box: QGroupBox
    _view: QTableView
    _txt_var: dict[str, QLineEdit] = {}
    _lists: dict[str, Union[QListWidget, QComboBox]] = {}

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

        self._lists['data'] = QListWidget()
        self._lists['data'].setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self._lists['data'].currentTextChanged.connect(self._refresh_pandas)

        self._lists['type'] = QComboBox()
        self._lists['type'].addItem('Crypto Currencies Daily')
        self._lists['type'].addItem('Standard Time Series Daily Adjusted')

        self._refresh_lists()

        validator = QRegularExpressionValidator(r'^[\w\-. ]+$')

        dataset_label = QLabel("Avaliable Datasets:")
        search_type_label = QLabel("Symbol/Search Type:")
        search_label = QLabel("Symbol/Search Term:")
        name_label = QLabel("Dataset Name:")

        left_options.addWidget(dataset_label, 0, 0)
        left_options.addWidget(self._lists['data'], 1, 0)

        right_options.addWidget(search_type_label, 0, 0)
        right_options.addWidget(self._lists['type'], 1, 0)

        self._txt_var['ds_name'] = QLineEdit()
        self._txt_var['data_search'] = QLineEdit()
        self._txt_var['ds_name'].setValidator(validator)
        self._txt_var['data_search'].setValidator(validator)

        right_options.addWidget(search_label, 2, 0)
        right_options.addWidget(self._txt_var['data_search'], 3, 0)

        right_options.addWidget(name_label, 5, 0)
        right_options.addWidget(self._txt_var['ds_name'], 6, 0)

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

    def _ingest(self) -> None:
        self.setEnabled(False)
        search_type = self._lists['type'].currentText()
        search_term = self._txt_var['data_search'].text()
        if search_type == 'Crypto Currencies Daily':
            try:
                data = data_ingest.async_get([data_ingest.get_cc_daily(
                    search_term, self.config, cache=False)])
                search_type = data_ingest.AVDataTypes.CryptoCurrenciesDaily
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
                search_type = data_ingest.AVDataTypes.TimeSeriesDailyAdjusted
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
        name = self._txt_var['ds_name'].text()
        if name == '':
            name = data_ingest.name_generator(
                search_term, search_type)
        else:
            name = data_ingest.name_generator(
                search_term, search_type, name)
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
        name = self._lists['data'].selectedItems()[0].text()

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
        'extra-imports': ['__future__', 'typing', 'PySide6', 'PySide6.QtGui',
                          'PySide6.QtWidgets', 'data_ingest', 'interface_pandas',
                          'configuration'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E1136']
    })
