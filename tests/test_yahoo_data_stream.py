import pandas as pd
import pytest
from unittest.mock import patch

from agentic_ai_system.yahoo_data_stream import YahooDataStream


@pytest.fixture
def yahoo_config():
    return {
        'data_source': {'type': 'yahoo'},
        'yahoo': {'poll_interval_seconds': 1, 'auto_adjust': False},
        'trading': {
            'symbol': 'AAPL',
            'timeframe': '1d',
        },
        'realtime_data': {'buffer_size': 10},
    }


def _sample_yahoo_frame():
    idx = pd.date_range('2024-06-03', periods=3, freq='D', tz='America/New_York')
    return pd.DataFrame(
        {
            'Open': [190.0, 191.0, 192.0],
            'High': [191.5, 192.5, 193.5],
            'Low': [189.0, 190.0, 191.0],
            'Close': [191.0, 192.0, 193.0],
            'Volume': [1_000_000, 1_100_000, 1_200_000],
        },
        index=idx,
    )


class TestYahooDataStream:
    def test_initialization_from_symbol(self, yahoo_config):
        stream = YahooDataStream(yahoo_config)
        assert stream.symbols == ['AAPL']
        assert stream.interval == '1d'

    def test_normalize_ohlcv(self, yahoo_config):
        stream = YahooDataStream(yahoo_config)
        df = stream._normalize_ohlcv(_sample_yahoo_frame())
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert len(df) == 3
        assert df['close'].iloc[-1] == 193.0

    def test_clamp_intraday_lookback(self, yahoo_config):
        yahoo_config['trading']['timeframe'] = '1m'
        stream = YahooDataStream(yahoo_config)
        start, end = stream._clamp_window('2020-01-01', '2026-01-01', '1m')
        assert start > '2020-01-01'
        assert end >= start

    def test_get_historical_data(self, yahoo_config):
        stream = YahooDataStream(yahoo_config)
        with patch.object(stream, '_download', return_value=_sample_yahoo_frame()):
            df = stream.get_historical_data('AAPL', '2024-01-01', '2024-12-31')
        assert len(df) == 3
        assert 'open' in df.columns
