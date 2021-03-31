import pytest
import pandas as pd
import py
import data_ingest

import os


class TestDataIngest():
    """PyTests for the data_ingest module."""

    def test_save_load(self, tmpdir: py._path.local.LocalPath):
        """Test saving and loading dataframes"""
        print(type(tmpdir))
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(d)
        data_ingest.save_data('test_file', df, location=tmpdir)
        assert df.equals(data_ingest.load_data('test_file', location=tmpdir))
