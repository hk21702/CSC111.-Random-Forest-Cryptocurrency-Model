import sys

from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from configuration import Config
import initialization
import interface as ui


def run(config: Config):
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_purple.xml')

    ex = ui.MainWindow(config)

    sys.exit(app.exec_())


if __name__ == '__main__':
    initialization.create_project_dirs()
    config = Config()
    run(config)
