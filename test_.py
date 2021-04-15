"""Module for pytests."""
import pytest
import pandas as pd
import py
import data_ingest


class TestDataIngest():
    """PyTests for the data_ingest module."""

    def test_save_load(self, tmpdir: py._path.local.LocalPath) -> None:
        """Test saving and loading dataframes"""
        print(type(tmpdir))
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(d)
        data_ingest.save_data('test_file', df, location=tmpdir)
        assert df.equals(data_ingest.load_data('test_file', location=tmpdir))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['pytest', 'pandas', 'py', 'data_ingest'],
        'allowed-io': ['test_save_load'],
        'max-line-length': 100,
        'disable': ['E1136']
    })
