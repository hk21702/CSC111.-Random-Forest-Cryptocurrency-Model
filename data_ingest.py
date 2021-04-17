"""Model containing functions related to ingest of new data, as well as
processing for usage in training and prediction input."""
from __future__ import annotations
import os
from collections import defaultdict
from datetime import datetime
from enum import Enum, auto

import asyncio
from asyncio.coroutines import coroutine

import pandas as pd
from alpha_vantage.async_support.cryptocurrencies import CryptoCurrencies
from alpha_vantage.async_support.timeseries import TimeSeries
from pandas.core.frame import DataFrame

import initialization
import common_exceptions as ce
from data_classes import WindowArgs, DataPair
from configuration import Config

SAVE_LOCATION = 'cache/data/'
MODEL_LOCATION = 'cache/models/'


class IngestTypes(Enum):
    """Valid api ingest types."""
    TimeSeriesDailyAdjusted = 'Time Series Daily Adjusted'
    CryptoCurrenciesDaily = 'Cryptocurrencies Daily'
    GoogleTrends = "Google Trends"

    @staticmethod
    def from_str(label: str) -> IngestTypes:
        if label in ('Time Series Daily Adjusted'):
            return IngestTypes.TimeSeriesDailyAdjusted
        elif label in ('Cryptocurrencies Daily'):
            return IngestTypes.CryptoCurrenciesDaily
        elif label in ('Google Trends'):
            return IngestTypes.GoogleTrends
        else:
            raise NotImplementedError


def load_data(name: str, location: str = SAVE_LOCATION) -> pd.DataFrame:
    """Returns saved pandas dataframe from a feather file with market data"""
    df = pd.read_feather(location + name + '.feather')
    if 'date' in df.columns.values:
        df = df.set_index('date')
    return df


def save_data(name: str, dataframe: pd.DataFrame, location: str = SAVE_LOCATION) -> None:
    """Saves pandas dataframe as a feather file."""
    if dataframe.index.dtype.kind == 'M':
        dataframe = dataframe.reset_index()
    try:
        dataframe.to_feather(location + name + '.feather')
    except FileNotFoundError:
        initialization.create_folder(location)
        dataframe.to_feather(location + name + '.feather')


def delete_data(name: str, location: str = None, extension: str = None,
                file_type: str = None) -> None:
    """Deletes a file

    Representation Invariants:
       - type in {'data', 'model'} or type is None
    """
    if file_type == 'data':
        location = SAVE_LOCATION
        extension = '.feather'
    elif file_type == 'model':
        location = MODEL_LOCATION
        extension = '.p'

    os.remove(location + name + extension)


def get_avaliable_data(location: str = None, extension: str = None,
                       search_type: str = None) -> list[str]:
    """Returns a list with all avaliable files in a folder.

       Representation Invariants:
       - search_type in {'data', 'model'} or type is None
    """
    if search_type == 'data':
        location = SAVE_LOCATION
        extension = '.feather'
    elif search_type == 'model':
        location = MODEL_LOCATION
        extension = '.p'
    files = os.listdir(location)
    return [file.replace(extension, '') for file in files if file.endswith(extension)]


def get_avaliable_sym(location: str = SAVE_LOCATION, extension: str = '.feather') -> set[str]:
    """Returns a list with the avaliable symbols or terms files in a folder."""
    files = get_avaliable_data(location, extension)
    return {sym.split(';', maxsplit=1)[0] for sym in files}


async def get_ts_daily_adjusted(symbol: str, config: Config, cache: bool = True) -> pd.DataFrame:
    """Returns time series data for the given symbol using the Alpha Vantage api in an
    async method."""
    ts = TimeSeries(key=config.get('key', 'ALPHA_VANTAGE'),
                    output_format='pandas')
    try:
        data, _ = await ts.get_daily_adjusted(symbol, outputsize='full')
    except ValueError as e:
        if 'higher API call' in str(e):
            raise ce.RateLimited
        elif 'Invalid API call' in str(e):
            raise ce.UnknownAVType
        else:
            raise
    idx = pd.date_range(min(data.index), max(data.index))
    data = data.reindex(idx[::-1])
    data['4. close'].fillna(method='backfill', inplace=True)
    data['5. adjusted close'].fillna(method='backfill', inplace=True)
    data['6. volume'].fillna(0.0, inplace=True)
    data['7. dividend amount'].fillna(0.0, inplace=True)
    data.apply(lambda x: x.fillna(data['4. close'], inplace=True)
               if x.name in ['1. open',
                             '2. high',
                             '3. low']
               else x.fillna(1.0, inplace=True))
    data.index.name = 'date'
    data = meta_label_columns(data, symbol)
    if cache:
        save_data(name_generator(
            symbol, IngestTypes.TimeSeriesDailyAdjusted), data)
    await ts.close()
    return data


async def get_cc_daily(symbol: str, config: Config, market: str = 'USD',
                       sanitize: bool = True, cache: bool = True) -> pd.DataFrame:
    """Returns CryptoCurrency data for the given symbol using the Alpha Vantage api in an
    async method."""
    cc = CryptoCurrencies(key=config.get('key', 'ALPHA_VANTAGE'),
                          output_format='pandas')
    try:
        data, _ = await cc.get_digital_currency_daily(symbol, market)
    except ValueError as e:
        if 'higher API call' in str(e):
            raise ce.RateLimited
        elif 'Invalid API call' in str(e):
            raise ce.UnknownAVType
        else:
            raise
    if sanitize:
        data = data[~(data.index >= datetime.now().date().strftime('%Y%m%d'))]
        cols = [x for x in data.columns if 'b. ' in x]
        data = data.drop(cols, axis=1)
    data = meta_label_columns(data, symbol)
    if cache:
        save_data(name_generator(
            symbol, IngestTypes.CryptoCurrenciesDaily), data)
    await cc.close()
    return data


def async_get(tasks: list[coroutine]) -> list[pd.DataFrame]:
    """Async handler for alpha vantage data get calls.
    """
    loop = asyncio.get_event_loop()
    group1 = asyncio.gather(*tasks)
    results = loop.run_until_complete(group1)
    return results


def name_generator(symbol: str, search_type: IngestTypes, name: str = None) -> str:
    """Returns autogenerated filename for auto cache save purposes."""
    now = datetime.now().strftime("%y-%d-%m_%H%M%S")
    if name is None:
        return f'{symbol};{search_type.name}_{now}'
    else:
        return f'{symbol};{search_type.name}_{name}'


def create_input(window_size: int, target_shift: int,
                 target_date: datetime, dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Creates a input window for predicting a specific date based on model requirements."""
    df = create_grouped_dataframe(dfs)
    window_end = target_date - pd.DateOffset(days=target_shift)
    window_start = window_end - pd.DateOffset(days=window_size - 1)
    df = df.loc[window_end: window_start]
    if df.shape[0] == window_size:
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        flat_win = df.stack()
        flat_win.index = flat_win.index.map('{0[1]}_{0[0]}'.format)
        return DataFrame().append(flat_win, ignore_index=True)
    else:
        raise ce.MissingData


def create_training_input(window: WindowArgs) -> DataPair:
    """Returns a dataset containing a pair of pandas dataframes that can
    be used for supervised learning."""
    df = create_grouped_dataframe(window.data_frames)
    x_train = DataFrame()
    y_train = DataFrame()
    for win in df.rolling(window.window_size, axis=1):
        if win.shape[0] == window.window_size:
            recent = win.head(1).index
            target_date = recent + pd.DateOffset(days=window.target_shift)
            if target_date[0] in window.target.index:
                win = win.reset_index(drop=True)
                win.index = win.index + 1
                flat_win = win.stack()
                flat_win.index = flat_win.index.map('{0[1]}_{0[0]}'.format)
                x_train = x_train.append(flat_win, ignore_index=True)
                y_train = y_train.append(
                    window.target.loc[target_date], ignore_index=True)
    return DataPair(x_train, y_train)


def create_grouped_dataframe(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Returns a dataframe where all similar symbols are merged by extending avaliable rows
    and then concatenated into a single dataframe."""
    groups = defaultdict(list)
    for obj in dfs:
        groups[tuple(map(tuple, obj.columns.values))].append(obj)

    group = list(groups.values())
    grouped_dfs = []

    for grouping in group:
        df = grouping[0]
        for frame in grouping[1:]:
            df = df.combine_first(frame)
        grouped_dfs.append(df)

    return pd.concat(grouped_dfs, join='inner', axis=1)


def meta_label_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Returns pandas DataFrame with renamed columns such that the dataframe
    name is a prefix for all columns."""
    cols = df.columns
    df_new = df.rename(columns={c: f'{name}; {c}' for c in cols})
    return df_new


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['numpy', 'asyncio', 'asyncio.coroutines', 'pandas',
                          'data_tools', 'tqdm.auto', 'pandas.core', 'pandas.core.frame',
                          'alpha_vantage.async_support.timeseries',
                          'os', 'enum', 'datetime', 'initialization',
                          'configuration', 'data_classes', 'pickle',
                          'alpha_vantage.async_support.cryptocurrencies',
                          'collections', 'common_exceptions'],
        'allowed-io': ['load_data', 'save_data', 'delete_data'],
        'max-line-length': 100,
        'disable': ['E1136']
    })
