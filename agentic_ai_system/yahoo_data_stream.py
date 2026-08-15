import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_INTERVAL_MAP = {
    '1min': '1m',
    '1m': '1m',
    '5min': '5m',
    '5m': '5m',
    '15min': '15m',
    '15m': '15m',
    '30min': '30m',
    '30m': '30m',
    '1H': '1h',
    '1h': '1h',
    '60min': '1h',
    '1D': '1d',
    '1d': '1d',
    '1day': '1d',
}

# Yahoo lookback limits by interval. Requesting more returns empty or errors.
_MAX_LOOKBACK = {
    '1m': pd.Timedelta(days=7),
    '2m': pd.Timedelta(days=60),
    '5m': pd.Timedelta(days=60),
    '15m': pd.Timedelta(days=60),
    '30m': pd.Timedelta(days=60),
    '60m': pd.Timedelta(days=730),
    '90m': pd.Timedelta(days=60),
    '1h': pd.Timedelta(days=730),
    '1d': None,
    '5d': None,
    '1wk': None,
    '1mo': None,
    '3mo': None,
}


class YahooDataStream:
    """
    Market data from Yahoo Finance via yfinance.

    Yahoo has no public equities WebSocket. This polls OHLCV bars.
    Quotes are typically delayed (~15 minutes for US equities).
    Unofficial API: rate limits and schema changes are expected failure modes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        trading = config.get('trading', {})
        if trading.get('symbols'):
            self.symbols = list(trading['symbols'])
        elif trading.get('symbol'):
            self.symbols = [trading['symbol']]
        else:
            self.symbols = ['AAPL']
        yahoo_cfg = config.get('yahoo', {})
        self.poll_interval = int(yahoo_cfg.get('poll_interval_seconds', 60))
        self.auto_adjust = bool(yahoo_cfg.get('auto_adjust', False))
        self.interval = self._map_interval(config.get('trading', {}).get('timeframe', '1d'))
        self.data_callbacks: List[Callable] = []
        self.is_connected = False
        self.data_buffer: Dict[str, Dict[str, Any]] = {}
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._last_bar_ts: Dict[str, pd.Timestamp] = {}
        self._buffer_size = int(config.get('realtime_data', {}).get('buffer_size', 100))

        for symbol in self.symbols:
            self.data_buffer[symbol] = {
                'trades': [],
                'quotes': [],
                'bars': [],
                'latest_bar': None,
            }

        logger.info(
            "Initialized YahooDataStream symbols=%s interval=%s poll_interval=%ss",
            self.symbols,
            self.interval,
            self.poll_interval,
        )

    @staticmethod
    def _map_interval(timeframe: str) -> str:
        mapped = _INTERVAL_MAP.get(str(timeframe), None)
        if mapped is None:
            logger.warning("Unknown timeframe %s, defaulting to 1d", timeframe)
            return '1d'
        return mapped

    def connect(self) -> None:
        """Start polling Yahoo for new bars."""
        if self.is_connected:
            logger.info("Yahoo data stream already connected")
            return

        self._stop_event.clear()
        self._poll_once()
        self._poll_thread = threading.Thread(target=self._poll_loop, name='yahoo-poll', daemon=True)
        self._poll_thread.start()
        self.is_connected = True
        logger.info("Yahoo data stream polling started")

    def disconnect(self) -> None:
        self._stop_event.set()
        self.is_connected = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=min(5, self.poll_interval + 1))
        logger.info("Disconnected from Yahoo data stream")

    def is_streaming(self) -> bool:
        return self.is_connected and self._poll_thread is not None and self._poll_thread.is_alive()

    def add_data_callback(self, callback: Callable) -> None:
        self.data_callbacks.append(callback)

    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.data_buffer:
            return {}
        buffer = self.data_buffer[symbol]
        return {
            'latest_trade': buffer['trades'][-1] if buffer['trades'] else None,
            'latest_quote': buffer['quotes'][-1] if buffer['quotes'] else None,
            'latest_bar': buffer['latest_bar'],
            'recent_trades': buffer['trades'][-10:] if buffer['trades'] else [],
            'recent_quotes': buffer['quotes'][-10:] if buffer['quotes'] else [],
        }

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        start, end = self._clamp_window(start_date, end_date, self.interval)
        try:
            raw = self._download(symbol, start=start, end=end, interval=self.interval)
            df = self._normalize_ohlcv(raw)
            if df.empty:
                logger.warning("No Yahoo historical data for %s between %s and %s", symbol, start, end)
            else:
                logger.info("Loaded %s Yahoo bars for %s (%s to %s)", len(df), symbol, start, end)
            return df
        except Exception as e:
            logger.error("Error fetching Yahoo historical data for %s: %s", symbol, e, exc_info=True)
            return pd.DataFrame()

    def test_connection(self) -> bool:
        try:
            symbol = self.symbols[0] if self.symbols else 'AAPL'
            df = self._download(symbol, period='5d', interval='1d')
            if df is None or df.empty:
                logger.warning("Yahoo connection test returned no data for %s", symbol)
                return False
            logger.info("Yahoo connection test succeeded for %s (%s rows)", symbol, len(df))
            return True
        except Exception as e:
            logger.error("Yahoo connection test failed: %s", e)
            return False

    def get_connection_status(self) -> Dict[str, Any]:
        return {
            'is_connected': self.is_connected,
            'provider': 'yahoo',
            'interval': self.interval,
            'poll_interval_seconds': self.poll_interval,
            'symbols': self.symbols,
            'data_buffers': {
                symbol: len(buffer['bars']) for symbol, buffer in self.data_buffer.items()
            },
        }

    def generate_simulated_data(self, symbol: str) -> Dict[str, Any]:
        import random

        latest_data = self.get_latest_data(symbol)
        base_price = 150.0
        if latest_data.get('latest_bar'):
            base_price = latest_data['latest_bar']['close']
        elif latest_data.get('latest_trade'):
            base_price = latest_data['latest_trade']['price']

        price_change = random.uniform(-0.01, 0.01) * base_price
        new_price = base_price + price_change
        simulated_bar = {
            'symbol': symbol,
            'open': base_price,
            'high': max(base_price, new_price),
            'low': min(base_price, new_price),
            'close': new_price,
            'volume': random.randint(100, 1000),
            'timestamp': int(time.time() * 1_000_000),
        }
        self._store_bar(symbol, simulated_bar, emit=False)
        return simulated_bar

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            try:
                self._poll_once()
            except Exception as e:
                logger.error("Yahoo poll loop error: %s", e, exc_info=True)

    def _poll_once(self) -> None:
        for symbol in self.symbols:
            try:
                raw = self._download(symbol, period='5d', interval=self.interval)
                df = self._normalize_ohlcv(raw)
                if df.empty:
                    logger.warning("Yahoo poll returned no bars for %s", symbol)
                    continue
                self._ingest_new_bars(symbol, df)
            except Exception as e:
                logger.error("Yahoo poll failed for %s: %s", symbol, e)

    def _ingest_new_bars(self, symbol: str, df: pd.DataFrame) -> None:
        last_ts = self._last_bar_ts.get(symbol)
        rows = df
        if last_ts is not None:
            rows = df[df['timestamp'] > last_ts]
        if rows.empty:
            return

        for _, row in rows.iterrows():
            ts = pd.Timestamp(row['timestamp'])
            bar = {
                'symbol': symbol,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'timestamp': int(ts.timestamp() * 1_000_000),
            }
            self._store_bar(symbol, bar, emit=True)
            self._last_bar_ts[symbol] = ts

    def _store_bar(self, symbol: str, bar: Dict[str, Any], emit: bool) -> None:
        buffer = self.data_buffer[symbol]
        buffer['bars'].append(bar)
        buffer['latest_bar'] = bar
        if len(buffer['bars']) > self._buffer_size:
            buffer['bars'] = buffer['bars'][-self._buffer_size:]
        if emit:
            self._notify_callbacks('bar', bar)

    def _notify_callbacks(self, data_type: str, data: Dict[str, Any]) -> None:
        for callback in self.data_callbacks:
            try:
                callback(data_type, data)
            except Exception as e:
                logger.error("Error in data callback: %s", e)

    def _clamp_window(self, start_date: str, end_date: str, interval: str) -> tuple:
        start = pd.to_datetime(start_date, utc=True).tz_convert(None)
        end = pd.to_datetime(end_date, utc=True).tz_convert(None)
        max_lookback = _MAX_LOOKBACK.get(interval)
        if max_lookback is not None:
            earliest = pd.Timestamp.now(tz='UTC').tz_convert(None) - max_lookback
            if start < earliest:
                logger.warning(
                    "Yahoo %s bars only cover ~%s; clamping start from %s to %s",
                    interval,
                    max_lookback,
                    start.date(),
                    earliest.date(),
                )
                start = earliest
        if end < start:
            end = start + pd.Timedelta(days=1)
        return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

    def _download(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
        interval: str = '1d',
    ) -> pd.DataFrame:
        import yfinance as yf

        kwargs: Dict[str, Any] = {
            'tickers': symbol,
            'interval': interval,
            'auto_adjust': self.auto_adjust,
            'progress': False,
            'threads': False,
        }
        if period:
            kwargs['period'] = period
        else:
            kwargs['start'] = start
            kwargs['end'] = end
        return yf.download(**kwargs)

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [str(col[0]).strip().lower() for col in out.columns]
        else:
            out.columns = [str(c).strip().lower() for c in out.columns]
        rename = {}
        if 'datetime' in out.columns:
            rename['datetime'] = 'timestamp'
        out = out.rename(columns=rename)

        if 'timestamp' not in out.columns:
            out = out.reset_index()
            time_col = out.columns[0]
            out = out.rename(columns={time_col: 'timestamp'})

        out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True).dt.tz_localize(None)

        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in out.columns]
        if missing:
            logger.error("Yahoo response missing columns: %s", missing)
            return pd.DataFrame(columns=required)

        out = out[required].dropna()
        out = out.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        return out.reset_index(drop=True)
