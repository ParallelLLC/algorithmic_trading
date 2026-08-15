import logging
import random
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

# How long each bar covers. Used to tell a finished bar from the one still
# forming right now -- see _drop_incomplete.
_INTERVAL_DURATION = {
    '1m': pd.Timedelta(minutes=1),
    '2m': pd.Timedelta(minutes=2),
    '5m': pd.Timedelta(minutes=5),
    '15m': pd.Timedelta(minutes=15),
    '30m': pd.Timedelta(minutes=30),
    '60m': pd.Timedelta(hours=1),
    '90m': pd.Timedelta(minutes=90),
    '1h': pd.Timedelta(hours=1),
    '1d': pd.Timedelta(days=1),
    '5d': pd.Timedelta(days=5),
    '1wk': pd.Timedelta(weeks=1),
}

# A daily-or-slower bar that moves more than this is almost always an
# unadjusted split rather than a real move (NVDA's 2024 10:1 shows up as -90%).
_SPLIT_SUSPECT_MOVE = 0.35


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
        # Adjusted by default. With auto_adjust off, Yahoo returns raw Close and
        # every split reads as a crash: NVDA's June 2024 10:1 becomes a -90% bar.
        self.auto_adjust = bool(yahoo_cfg.get('auto_adjust', True))
        self.emit_incomplete_bars = bool(yahoo_cfg.get('emit_incomplete_bars', False))
        self.max_backoff = int(yahoo_cfg.get('max_backoff_seconds', 900))
        self.interval = self._map_interval(config.get('trading', {}).get('timeframe', '1d'))
        self._consecutive_failures = 0
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

        if not self.auto_adjust:
            logger.warning(
                "yahoo.auto_adjust is false: prices are NOT split- or dividend-adjusted. "
                "Every split will appear as a large single-bar loss and any backtest "
                "spanning one will be wrong."
            )

        logger.info(
            "Initialized YahooDataStream symbols=%s interval=%s poll_interval=%ss "
            "auto_adjust=%s emit_incomplete_bars=%s",
            self.symbols,
            self.interval,
            self.poll_interval,
            self.auto_adjust,
            self.emit_incomplete_bars,
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
        # Seed the backoff from the first attempt: if we are already being
        # throttled, the loop should start backed off rather than hammering.
        self._consecutive_failures = 0 if self._poll_once() else 1
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
            df = self._drop_incomplete(self._normalize_ohlcv(raw))
            self._warn_if_unadjusted(symbol, df)
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
        delay = self.poll_interval
        while not self._stop_event.wait(delay):
            try:
                succeeded = self._poll_once()
            except Exception as e:
                logger.error("Yahoo poll loop error: %s", e, exc_info=True)
                succeeded = False
            self._consecutive_failures = 0 if succeeded else self._consecutive_failures + 1
            delay = self._next_delay()

    def _next_delay(self) -> float:
        """Poll interval, backed off exponentially while Yahoo is refusing us.

        Yahoo rate-limits aggressively and an unofficial API gives no
        Retry-After, so a fixed interval just keeps you throttled. Jitter stops
        several symbols (or several deployments) resynchronising after an outage.
        """
        if self._consecutive_failures == 0:
            base = float(self.poll_interval)
        else:
            base = min(
                self.poll_interval * (2 ** self._consecutive_failures),
                float(self.max_backoff),
            )
            logger.warning(
                "Yahoo poll failed %s time(s) in a row; next attempt in ~%.0fs",
                self._consecutive_failures,
                base,
            )
        return max(1.0, base * random.uniform(0.8, 1.2))

    def _poll_once(self) -> bool:
        """Fetch and ingest one round of bars. Returns True if any symbol succeeded."""
        any_success = False
        for symbol in self.symbols:
            try:
                raw = self._download(symbol, period='5d', interval=self.interval)
                df = self._normalize_ohlcv(raw)
                if df.empty:
                    logger.warning("Yahoo poll returned no bars for %s", symbol)
                    continue
                self._ingest_new_bars(symbol, df)
                any_success = True
            except Exception as e:
                logger.error("Yahoo poll failed for %s: %s", symbol, e)
        return any_success

    def _warn_if_unadjusted(self, symbol: str, df: pd.DataFrame) -> int:
        """Flag single-bar moves that look like unadjusted corporate actions.

        This is a backstop rather than the fix -- the fix is auto_adjust. But a
        split slipping through silently corrupts every downstream number, so it
        is worth naming the dates rather than letting a strategy trade them.
        Returns the number of suspicious bars found.
        """
        duration = _INTERVAL_DURATION.get(self.interval)
        if df.empty or len(df) < 2 or duration is None or duration < pd.Timedelta(days=1):
            return 0
        moves = df['close'].pct_change()
        suspects = df.loc[moves.abs() > _SPLIT_SUSPECT_MOVE, 'timestamp']
        if len(suspects):
            dates = ', '.join(str(pd.Timestamp(t).date()) for t in suspects.head(5))
            logger.warning(
                "%s has %s bar(s) moving more than %.0f%% (%s). On a liquid name that is "
                "usually an unadjusted split, not a real move — check yahoo.auto_adjust.",
                symbol,
                len(suspects),
                _SPLIT_SUSPECT_MOVE * 100,
                dates,
            )
        return int(len(suspects))

    def _drop_incomplete(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove the bar that is still forming.

        Yahoo returns the in-progress period as an ordinary row. Emitting it
        would hand the strategy a close that has not happened yet, and because
        the watermark advances past it, the finished version never arrives.
        """
        if self.emit_incomplete_bars or df.empty:
            return df
        duration = _INTERVAL_DURATION.get(self.interval)
        if duration is None:
            return df
        now = pd.Timestamp.now(tz='UTC').tz_convert(None)
        complete = df[df['timestamp'] + duration <= now]
        dropped = len(df) - len(complete)
        if dropped:
            logger.debug("Dropped %s in-progress %s bar(s)", dropped, self.interval)
        return complete

    def _ingest_new_bars(self, symbol: str, df: pd.DataFrame) -> None:
        rows = self._drop_incomplete(df)
        last_ts = self._last_bar_ts.get(symbol)
        if last_ts is not None:
            rows = rows[rows['timestamp'] > last_ts]
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
