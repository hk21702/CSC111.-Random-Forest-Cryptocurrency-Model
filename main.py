from pandas import DataFrame, Series

from configuration import Config
import random_forest
import data_ingest
import data_tools
import initialization
from data_classes import WindowArgs, ForestArgs
import reg_tree

if __name__ == '__main__':
    initialization.create_project_dirs()
    config = Config()
    df2 = data_ingest.load_data('AMC_TimeSeriesDailyAdjusted_21-02-04_170438')
    df1 = data_ingest.load_data('BTC_CryptoCurrenciesDaily_21-02-04_170438')
    target = DataFrame(df1['BTC 4a. close (USD)'])
    # x_train, y_train = data_ingest.create_training_input(
    #    60, [df1, df2], target, 14)
    inputs = data_ingest.create_input(60, [df1, df2])
    window = WindowArgs(60, 14, [df1, df2], target)
    forest_params = ForestArgs(10, 350)
   # a, b = data_ingest.create_training_input(30, [df1, df2], target, 7)
    model = random_forest.RandomForest(
        window, forest_params)
    a = model.predict(inputs.iloc[[0, 1, 2]])
    print(a)
    # initialization.run()
