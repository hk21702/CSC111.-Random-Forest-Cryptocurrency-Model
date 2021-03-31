""" Module for all initialization proceses related
to data ingest and initial pre-processing.
"""

import pandas as pd
import sys
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet
from pprint import pprint

import interface as ui


def run():
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_purple.xml')

    ex = ui.MainWindow()

    sys.exit(app.exec_())
