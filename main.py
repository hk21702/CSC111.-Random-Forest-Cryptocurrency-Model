import sys

from pandas import DataFrame, Series
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from configuration import Config
import random_forest
import data_ingest
import data_tools
import initialization
from data_classes import WindowArgs, ForestArgs
import reg_tree
import interface as ui

from datetime import datetime


def run(config: Config):
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_purple.xml')

    ex = ui.MainWindow(config)

    sys.exit(app.exec_())


if __name__ == '__main__':
    initialization.create_project_dirs()
    config = Config()
    #df1 = data_ingest.load_data(
    #    'BTC_CryptoCurrenciesDaily_21-15-04_002917')
    # df2 = data_ingest.load_data(
    #    'TSLA_TimeSeriesDailyAdjusted_21-15-04_082902')
    #df3 = data_ingest.load_data('TSLA_TimeSeriesDailyAdjusted_21-15-04_082902')
    #df5 = data_ingest.create_grouped_dataframe([df2])

    if False == True:
        target = DataFrame(df1['BTC 1a. open (USD)'])
        # x_train, y_train = data_ingest.create_training_input(
        #    60, [df1, df2], target, 14)
        now = datetime.now().date()
        #input = data_ingest.create_input(10, 20, now, [df1, df2])
        window = WindowArgs(60, 20, {'BTC', 'TSLA'}, [
                            df1, df2], target, 'BTC 1a. open (USD)')
        forest_params = ForestArgs(3, 300)
        #a, b = data_ingest.create_training_input(30, [df1, df2], target, 7)
        model = random_forest.RandomForest(
            window, forest_params)
        random_forest.save_model('wowers', model)
        ok = random_forest.load_model('wowers')
        test_dfs = random_forest.load_corresponding_dataframes(ok)
        input = data_ingest.create_input(60, 20, now, test_dfs)
        a = ok.predict(input)
        print(a)
    else:
        run(config)
