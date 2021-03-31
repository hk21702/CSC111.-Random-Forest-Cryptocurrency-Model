import sys

from configuration import Config

from PySide6 import QtWidgets, QtGui
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PySide6.QtGui import QCloseEvent


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setup()

    def setup(self):
        hbox = QtWidgets.QHBoxLayout(self)

        btn_train = QPushButton('Train Model', self)
        btn_train.resize(btn_train.sizeHint())
        self.training_window = TrainingWindow()

        btn_predict = QPushButton('Predict', self)
        btn_predict.resize(btn_predict.sizeHint())

        btn_quit = QPushButton('Force Quit', self)
        btn_quit.clicked.connect(QApplication.instance().quit)
        btn_quit.resize(btn_quit.sizeHint())

        hbox.addWidget(btn_train)
        hbox.addWidget(btn_predict)
        hbox.addWidget(btn_quit)

        self.setWindowTitle('Crypto Futures')
        self.show()
        self.setFixedSize(self.size())

    def new_window(self, window: QWidget) -> None:
        self.training_window.show

    def closeEvent(self, event: QCloseEvent):
        reply = QMessageBox.question(self, 'Message', 'Are you sure you want to quit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setup()

    def setup(self):
        hbox = QtWidgets.QHBoxLayout(self)

        btn_train = QPushButton('Train Model', self)
        btn_train.resize(btn_train.sizeHint())

        btn_predict = QPushButton('Predict', self)
        btn_predict.resize(btn_predict.sizeHint())

        btn_quit = QPushButton('Force Quit', self)
        btn_quit.clicked.connect(QApplication.instance().quit)
        btn_quit.resize(btn_quit.sizeHint())

        hbox.addWidget(btn_train)
        hbox.addWidget(btn_predict)
        hbox.addWidget(btn_quit)

        self.setWindowTitle('Train Model')
        self.setFixedSize(self.size())
