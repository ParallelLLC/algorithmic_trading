import logging

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


def _split_frame():
    """An unadjusted 10:1 split, as Yahoo returns it with auto_adjust=False.

    Modelled on NVDA, 10 June 2024: the raw Close drops from ~1200 to ~120 and
    a backtest reads it as a -90% day.
    """
    idx = pd.date_range('2024-06-06', periods=4, freq='D', tz='America/New_York')
    close = [1200.0, 1208.0, 120.5, 121.0]
    return pd.DataFrame(
        {
            'Open': close,
            'High': [c * 1.01 for c in close],
            'Low': [c * 0.99 for c in close],
            'Close': close,
            'Volume': [1_000_000] * 4,
        },
        index=idx,
    )


def _dated_frame(timestamps, close=100.0):
    return pd.DataFrame(
        {
            'timestamp': [pd.Timestamp(t) for t in timestamps],
            'open': close,
            'high': close,
            'low': close,
            'close': close,
            'volume': 1_000.0,
        }
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


class TestPriceAdjustment:
    """Unadjusted prices turn every split into a phantom crash."""

    def test_adjustment_is_on_by_default(self):
        stream = YahooDataStream({'trading': {'symbol': 'AAPL', 'timeframe': '1d'}})
        assert stream.auto_adjust is True

    def test_auto_adjust_is_passed_through_to_yfinance(self):
        stream = YahooDataStream({'trading': {'symbol': 'AAPL', 'timeframe': '1d'}})
        with patch('yfinance.download', return_value=_sample_yahoo_frame()) as download:
            stream._download('AAPL', period='5d', interval='1d')
        assert download.call_args.kwargs['auto_adjust'] is True

    def test_opting_out_of_adjustment_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            YahooDataStream({
                'trading': {'symbol': 'AAPL', 'timeframe': '1d'},
                'yahoo': {'auto_adjust': False},
            })
        assert any('not split' in r.message.lower() for r in caplog.records)

    def test_split_sized_move_is_flagged(self, yahoo_config, caplog):
        stream = YahooDataStream(yahoo_config)
        df = stream._normalize_ohlcv(_split_frame())
        with caplog.at_level(logging.WARNING):
            found = stream._warn_if_unadjusted('NVDA', df)
        assert found == 1
        assert any('auto_adjust' in r.message for r in caplog.records)

    def test_ordinary_moves_are_not_flagged(self, yahoo_config, caplog):
        stream = YahooDataStream(yahoo_config)
        df = stream._normalize_ohlcv(_sample_yahoo_frame())
        with caplog.at_level(logging.WARNING):
            assert stream._warn_if_unadjusted('AAPL', df) == 0

    def test_intraday_bars_are_not_split_checked(self, yahoo_config):
        """A 40% move in one minute is a halt or a fat finger, not a split."""
        yahoo_config['trading']['timeframe'] = '1m'
        stream = YahooDataStream(yahoo_config)
        df = stream._normalize_ohlcv(_split_frame())
        assert stream._warn_if_unadjusted('NVDA', df) == 0


class TestIncompleteBars:
    """The bar Yahoo is still building must not be reported as final."""

    def test_forming_bar_is_dropped(self, yahoo_config):
        stream = YahooDataStream(yahoo_config)
        now = pd.Timestamp.now(tz='UTC').tz_convert(None).normalize()
        df = _dated_frame([now - pd.Timedelta(days=2), now - pd.Timedelta(days=1), now])
        kept = stream._drop_incomplete(df)
        assert len(kept) == 2
        assert kept['timestamp'].max() < now

    def test_finished_bars_all_survive(self, yahoo_config):
        stream = YahooDataStream(yahoo_config)
        now = pd.Timestamp.now(tz='UTC').tz_convert(None).normalize()
        df = _dated_frame([now - pd.Timedelta(days=5), now - pd.Timedelta(days=4)])
        assert len(stream._drop_incomplete(df)) == 2

    def test_opting_in_keeps_the_forming_bar(self, yahoo_config):
        yahoo_config['yahoo']['emit_incomplete_bars'] = True
        stream = YahooDataStream(yahoo_config)
        now = pd.Timestamp.now(tz='UTC').tz_convert(None).normalize()
        df = _dated_frame([now - pd.Timedelta(days=1), now])
        assert len(stream._drop_incomplete(df)) == 2

    def test_partial_bar_is_never_emitted_then_stranded(self, yahoo_config):
        """The bug this guards: emitting the forming bar advanced the watermark,
        so the finished version of that same bar never reached a callback."""
        stream = YahooDataStream(yahoo_config)
        received = []
        stream.add_data_callback(lambda kind, bar: received.append(bar))

        now = pd.Timestamp.now(tz='UTC').tz_convert(None).normalize()
        yesterday, today = now - pd.Timedelta(days=1), now

        stream._ingest_new_bars('AAPL', _dated_frame([yesterday, today], close=100.0))
        assert len(received) == 1  # only yesterday's completed bar

        # Next day: what was the forming bar is now final and must arrive.
        with patch.object(stream, '_drop_incomplete', side_effect=lambda d: d):
            stream._ingest_new_bars('AAPL', _dated_frame([yesterday, today], close=105.0))
        assert len(received) == 2
        assert received[-1]['close'] == 105.0


class TestPollBackoff:
    """Yahoo rate-limits hard, and a fixed interval keeps you throttled."""

    def test_success_polls_at_the_configured_interval(self, yahoo_config):
        yahoo_config['yahoo']['poll_interval_seconds'] = 60
        stream = YahooDataStream(yahoo_config)
        stream._consecutive_failures = 0
        assert 48 <= stream._next_delay() <= 72  # 60s +/- jitter

    def test_delay_grows_with_consecutive_failures(self, yahoo_config):
        yahoo_config['yahoo']['poll_interval_seconds'] = 60
        stream = YahooDataStream(yahoo_config)
        delays = []
        for failures in (1, 2, 3):
            stream._consecutive_failures = failures
            delays.append(stream._next_delay())
        assert delays[0] < delays[1] < delays[2]

    def test_backoff_is_capped(self, yahoo_config):
        yahoo_config['yahoo']['poll_interval_seconds'] = 60
        yahoo_config['yahoo']['max_backoff_seconds'] = 300
        stream = YahooDataStream(yahoo_config)
        stream._consecutive_failures = 20
        assert stream._next_delay() <= 300 * 1.2

    def test_jitter_desynchronises_retries(self, yahoo_config):
        stream = YahooDataStream(yahoo_config)
        stream._consecutive_failures = 3
        assert len({stream._next_delay() for _ in range(20)}) > 1

    def test_poll_reports_failure_when_every_symbol_fails(self, yahoo_config):
        stream = YahooDataStream(yahoo_config)
        with patch.object(stream, '_download', side_effect=RuntimeError('429 Too Many Requests')):
            assert stream._poll_once() is False

    def test_poll_reports_success_when_a_symbol_returns_bars(self, yahoo_config):
        stream = YahooDataStream(yahoo_config)
        with patch.object(stream, '_download', return_value=_sample_yahoo_frame()):
            assert stream._poll_once() is True

    def test_empty_response_counts_as_failure(self, yahoo_config):
        """A rate-limited yfinance returns an empty frame rather than raising."""
        stream = YahooDataStream(yahoo_config)
        with patch.object(stream, '_download', return_value=pd.DataFrame()):
            assert stream._poll_once() is False
