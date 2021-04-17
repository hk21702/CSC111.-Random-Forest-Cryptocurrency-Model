""""The main file for the CSC111 term project.
See report for usage instructions. Tested on
Windows 10, Python 3.9.1, 64 bit.  
"""
import sys

from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from configuration import Config
import initialization
import interface as ui

if __name__ == '__main__':
    initialization.create_project_dirs()
    config = Config()
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_purple.xml')

    # Assigning the window to a variable seems to be required.
    ex = ui.MainWindow(config)

    sys.exit(app.exec_())
