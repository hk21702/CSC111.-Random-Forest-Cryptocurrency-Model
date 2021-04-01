import asyncio
from asyncio.coroutines import coroutine
from alpha_vantage.async_support.cryptocurrencies import CryptoCurrencies
from alpha_vantage.async_support.timeseries import TimeSeries
import pytrends
import pandas as pd
import numpy as np

import os
from datetime import datetime
from enum import Enum, auto

import initialization
from configuration import Config

SAVE_LOCATION = 'cache/data/'

MARKET_SYMBOLS = []
CRYPTO_SYMBOLS = []


class AVDataTypes(Enum):
    """Valid alpha vantage api call types."""
    TimeSeriesDailyAdjusted = auto()
    CryptoCurrenciesDaily = auto()


class UnknownAVType(Exception):
    """Exception raised when trying to handle an unknown type of Alpha Vantage
    API call."""


class RateLimited(Exception):
    """Exception raised when an API call fails because of a rate limit."""


def load_data(name: str, location: str = SAVE_LOCATION) -> pd.DataFrame:
    """Returns saved pandas dataframe from a feather file with market data"""
    df = pd.read_feather(location + name + '.feather')
    if 'date' in df.columns.values:
        df = df.set_index('date')
    return df


def save_data(name: str, dataframe: pd.DataFrame, location: str = SAVE_LOCATION) -> None:
    """Saves pandas dataframe as a feather file."""
    if dataframe.index.name == 'date':
        dataframe = dataframe.reset_index()
    try:
        dataframe.to_feather(location + name + '.feather')
    except FileNotFoundError:
        initialization.create_folder(location)
        dataframe.to_feather(location + name + '.feather')


def delete_data(name: str, location: str = SAVE_LOCATION, extension: str = '.feather') -> None:
    """Deletes a file"""
    os.remove(location + name + extension)


async def get_TS_daily_adjusted(symbol: str, config: Config, cache: bool = True) -> pd.DataFrame:
    """Returns time series data for the given symbol using the Alpha Vantage api in an
    async method."""
    ts = TimeSeries(key=config.get('key', 'ALPHA_VANTAGE'),
                    output_format='pandas')
    try:
        data, _ = await ts.get_daily_adjusted(symbol, outputsize='full')
    except ValueError as e:
        if 'higher API call' in str(e):
            raise RateLimited
        else:
            raise
    if cache:
        save_data(name_generator(
            symbol, AVDataTypes.TimeSeriesDailyAdjusted), data)
    await ts.close()
    return data


async def get_CC_daily(symbol: str, config: Config, market: str = 'USD', sanitize: bool = True, cache: bool = True) -> pd.DataFrame:
    """Returns CryptoCurrency data for the given symbol using the Alpha Vantage api in an
    async method."""
    cc = CryptoCurrencies(key=config.get('key', 'ALPHA_VANTAGE'),
                          output_format='pandas')
    try:
        data, _ = await cc.get_digital_currency_daily(symbol, market)
    except ValueError as e:
        if 'higher API call' in str(e):
            raise RateLimited
        else:
            raise
    if sanitize:
        data.drop(data.loc[datetime.now().date().strftime('%Y%m%d')].index)
        cols = [x for x in data.columns if 'b. ' in x]
        data = data.drop(cols, axis=1)
    if cache:
        save_data(name_generator(
            symbol, AVDataTypes.TimeSeriesDailyAdjusted), data)
    await cc.close()
    return data


def async_get(tasks: list[coroutine]) -> list[pd.DataFrame]:
    """Async handler for alpha vantage data get calls.
    """
    loop = asyncio.get_event_loop()
    group1 = asyncio.gather(*tasks)
    results = loop.run_until_complete(group1)
    return results


def name_generator(symbol: str, type: AVDataTypes) -> str:
    """Returns autogenerated filename for auto cache save purposes."""
    now = datetime.now().strftime("%y-%d-%m_%H%M%S")
    return '{symbol}_{type}_{time}'.format(symbol=symbol, type=type.name, time=now)


config = Config()
async_get([get_CC_daily(x, config) for x in ['BTC', 'ETH', 'DOGE']])
async_get([get_TS_daily_adjusted(x, config)
          for x in ['GME', 'TSM', 'AMC', 'APPL', 'TSLA']])


# TODO
"""daily percent change, meta data for data frames, merge data frames 
with renamed columns based on meta data, window dataframe, add reddit data for avaliable dates
get reddit data positivity score, get daily search interest, flatten window for model input"""
