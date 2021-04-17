"""This module contains the get_daily_trend function which obtains consecutive daily
trend data from google trends, and merges them in a way that normalizes the interest
values."""
from datetime import datetime, timedelta
from time import sleep

import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
from data_classes import DataPair
from data_ingest import meta_label_columns
import common_exceptions as ce


def get_daily_trend(trendreq: TrendReq, keyword: str, start: datetime,
                    end: datetime, cat: int = 0) -> pd.DataFrame:
    """Returns stitched and scaled consecutive daily trends data between start and end date.
    This function will first download piece-wise google trends data and then
    scale each piece using the overlapped period.
    """

    init_end = end
    init_end.replace(hour=23, minute=59, second=59)

    itr_d = end - timedelta(days=269)
    overlap_start = None

    frames = DataPair(pd.DataFrame(), pd.DataFrame())

    while end > start:
        tf = itr_d.strftime('%Y-%m-%d') + ' ' + end.strftime('%Y-%m-%d')

        temp = _fetch_data(
            trendreq, [keyword], timeframe=tf, cat=cat)
        temp.drop(columns=['isPartial'], inplace=True)
        temp.columns.values[0] = tf
        ol_temp = temp.copy()
        ol_temp.iloc[:, :] = None
        if overlap_start is not None:
            y1 = temp.loc[overlap_start:end].iloc[:, 0].values.max()
            y2 = frames.x.loc[overlap_start:end].iloc[:, -1].values.max()
            coef = y2 / y1
            temp = temp * coef
            ol_temp.loc[overlap_start:end, :] = 1

        frames.x = pd.concat([frames.x, temp], axis=1)
        frames.y = pd.concat([frames.y, ol_temp], axis=1)

        overlap_start = itr_d
        end -= (timedelta(days=269) - timedelta(days=100))
        itr_d -= (timedelta(days=269) - timedelta(days=100))
        # in case of short query interval getting banned by server
        sleep(0)

    frames = _fetch_seven_day(trendreq, keyword, init_end, frames, cat)

    frames = _merge_overlap(keyword, start, init_end, frames)

    frames.x.drop('overlap', axis=1, inplace=True)

    frames.x.index.name = 'date'

    return meta_label_columns(frames.x.iloc[::-1], f"{keyword} {cat}")


def _fetch_seven_day(trendreq: TrendReq, keyword: str, init_end: datetime,
                     frames: DataPair, cat: int) -> DataPair:
    """Returns a dataset after fetching seven final seven day data and scaling
    and merging it into the dual data fames in a data set."""
    frames.x.sort_index(inplace=True)
    frames.y.sort_index(inplace=True)
    if frames.x.index.max() < init_end:
        tf = 'now 7-d'
        hourly = _fetch_data(
            trendreq, [keyword], timeframe=tf, cat=cat)
        hourly.drop(columns=['isPartial'], inplace=True)

        # convert hourly data to daily data
        daily = hourly.groupby(hourly.index.date).sum()

        # check whether the first day data is complete (i.e. has 24 hours)
        daily['hours'] = hourly.groupby(hourly.index.date).count()
        if daily.iloc[0].loc['hours'] != 24:
            daily.drop(daily.index[0], inplace=True)
        daily.drop(columns='hours', inplace=True)

        daily.set_index(pd.DatetimeIndex(daily.index), inplace=True)
        daily.columns = [tf]

        ol_temp = daily.copy()
        ol_temp.iloc[:, :] = None
        # find the overlapping date
        intersect = frames.x.index.intersection(daily.index)

        # scaling use the overlapped today-4 to today-7 data
        coef = frames.x.loc[intersect].iloc[:, 0].max(
        ) / daily.loc[intersect].iloc[:, 0].max()
        daily = (daily * coef).round(decimals=0)
        ol_temp.loc[intersect, :] = 1

        frames.x = pd.concat([daily, frames.x], axis=1)
        frames.y = pd.concat([ol_temp, frames.y], axis=1)

    return frames


def _merge_overlap(keyword: str, start: datetime, init_end: datetime, frames: DataPair) -> DataPair:
    """Returns a dataset with a completed x set. Merges frames the two data frames
    and re-normalize the final dataframe"""

    # taking averages for overlapped period
    frames.x = frames.x.mean(axis=1)
    frames.y = frames.y.max(axis=1)
    # merge the two dataframe (trend data and overlap flag)
    frames.x = pd.concat([frames.x, frames.y], axis=1)
    frames.x.columns = [keyword, 'overlap']
    # Correct the timezone difference
    frames.x.index = frames.x.index + timedelta(minutes=0)
    frames.x = frames.x[start:init_end]
    # re-normalized to the overall maximum value to have max =100
    frames.x[keyword] = (100 * frames.x[keyword] / frames.x[keyword].max()
                         ).round(decimals=0)

    return frames


def _fetch_data(trendreq: TrendReq, kw_list: list[str], timeframe: str = 'today 3-m',
                cat: int = 0) -> pd.DataFrame:
    """Download google trends data using pytrends TrendReq and retries in
    case of a ResponseError."""
    attempts, fetched = 0, False
    while not fetched:
        try:
            trendreq.build_payload(
                kw_list=kw_list, timeframe=timeframe, cat=cat, geo='', gprop='')
        except ResponseError as e:
            print(e)
            print(f'Trying again in {60 + 5 * attempts} seconds.')
            sleep(60 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attempts, abort fetching.')
                raise ce.RateLimited
        else:
            fetched = True
    return trendreq.interest_over_time()


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['numpy', 'pandas', 'datetime', 'pytrends.exceptions',
                          'common_exceptions', 'pytrends.request', 'time', 'data_classes',
                          'data_ingest'],
        'allowed-io': ['_fetch_data'],
        'max-line-length': 100,
        'disable': ['E1136']
    })
