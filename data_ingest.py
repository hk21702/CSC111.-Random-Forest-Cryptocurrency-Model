import alpha_vantage as av
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import pandas as pd

import os

import initialization
from configuration import Config

SAVE_LOCATION = 'cache/data/'


def load_data(name: str, location: str = SAVE_LOCATION) -> pd.DataFrame:
    """Returns saved pandas dataframe from a feather file with market data"""
    return pd.read_feather(location + name + '.feather')


def save_data(name: str, dataframe: pd.DataFrame, location: str = SAVE_LOCATION) -> None:
    """Saves pandas dataframe as a feather file."""
    try:
        dataframe.to_feather(location + name + '.feather')
    except FileNotFoundError:
        initialization.create_folder(location)
        dataframe.to_feather(location + name + '.feather')


def delete_data(name: str, location: str = SAVE_LOCATION, extension: str = '.feather') -> None:
    """Deletes a file"""
    os.remove(location + name + extension)
