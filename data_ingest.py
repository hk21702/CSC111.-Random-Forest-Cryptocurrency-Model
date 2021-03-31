import alpha_vantage as av
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import pandas

from configuration import Config


def check_key(config: Config):
    cc = CryptoCurrencies(key='yea')
    

