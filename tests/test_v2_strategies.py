"""Strategy zoo, data loading, and the end-to-end Lab pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader import LabConfig, run_lab
from algotrader.data import _normalise, load_ohlcv, simulate_ohlcv
from algotrader.indicators import atr, bollinger, donchian, ema, macd, rsi, sma
from algotrader.lab import run_arena
from algotrader.strategies import REGISTRY, get_strategy, list_strategies
from algotrader.types import MarketData
from algotrader.verdict import reality_score

ALL_KEYS = sorted(REGISTRY)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return simulate_ohlcv("AAPL", "2016-01-01", "2023-01-01")


class TestIndicatorsAreCausal:
    """An indicator at bar t must not move when bars after t arrive."""

    @pytest.mark.parametrize(
        "fn",
        [
            lambda d: sma(d["close"], 20),
            lambda d: ema(d["close"], 12),
            lambda d: rsi(d["close"], 14),
            lambda d: macd(d["close"])[0],
            lambda d: bollinger(d["close"])[2],
            lambda d: atr(d, 14),
            lambda d: donchian(d, 20)[1],
        ],
    )
    def test_prefix_is_stable(self, bars, fn):
        cut = 800
        full = fn(bars).iloc[:cut]
        partial = fn(bars.iloc[:cut])
        pd.testing.assert_series_equal(full, partial, check_names=False, rtol=1e-9)

    def test_rsi_stays_in_range(self, bars):
        values = rsi(bars["close"], 14).dropna()
        assert values.between(0, 100).all()

    def test_donchian_excludes_the_current_bar(self, bars):
        _, upper = donchian(bars, 20)
        # A breakout must be possible: the channel cannot already contain today.
        assert (bars["high"] > upper).any()


class TestStrategies:
    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_output_is_well_formed(self, bars, key):
        target = get_strategy(key).generate(bars)
        assert target.index.equals(bars.index)
        assert target.notna().all()
        assert target.between(-1.0, 1.0).all()

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_signals_are_causal(self, bars, key):
        cut = 900
        full = get_strategy(key).generate(bars).iloc[:cut]
        partial = get_strategy(key).generate(bars.iloc[:cut])
        pd.testing.assert_series_equal(full, partial, check_names=False, rtol=1e-9)

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_grid_is_non_empty_and_valid(self, key):
        strategy = get_strategy(key)
        grid = strategy.grid(limit=40)
        assert 1 <= len(grid) <= 40
        for combo in grid:
            assert set(combo) <= {p.name for p in strategy.params}
            if "fast" in combo and "slow" in combo:
                assert combo["fast"] < combo["slow"]

    def test_unknown_strategy_names_the_alternatives(self):
        with pytest.raises(KeyError, match="Available"):
            get_strategy("does_not_exist")

    def test_clean_ignores_unknown_params_and_casts_types(self):
        strategy = get_strategy("sma_cross")
        cleaned = strategy.clean({"fast": 15.7, "nonsense": 1})
        assert cleaned == {"fast": 15, "slow": 100}

    def test_buy_and_hold_is_always_fully_invested(self, bars):
        assert (get_strategy("buy_and_hold").generate(bars) == 1.0).all()

    def test_list_strategies_honours_exclusions(self):
        keys = {s.key for s in list_strategies(exclude=["coin_flip"])}
        assert "coin_flip" not in keys and "sma_cross" in keys


class TestData:
    def test_simulation_is_deterministic_per_symbol(self):
        a = simulate_ohlcv("NVDA", "2018-01-01", "2022-01-01")
        b = simulate_ohlcv("NVDA", "2018-01-01", "2022-01-01")
        pd.testing.assert_frame_equal(a, b)

    def test_different_symbols_simulate_differently(self):
        a = simulate_ohlcv("NVDA", "2018-01-01", "2022-01-01")
        b = simulate_ohlcv("TSLA", "2018-01-01", "2022-01-01")
        assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())

    def test_simulated_bars_are_internally_consistent(self):
        df = simulate_ohlcv("SPY", "2015-01-01", "2023-01-01")
        assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
        assert (df["close"] > 0).all()
        assert df.index.is_monotonic_increasing

    def test_simulation_has_fat_tails_and_vol_clustering(self):
        """Naive GBM flatters strategies; the simulator must be harder than that."""
        returns = simulate_ohlcv("SPY", "2005-01-01", "2023-01-01")["close"].pct_change().dropna()
        assert returns.kurtosis() > 1.0
        assert returns.abs().autocorr(1) > 0.05

    def test_offline_load_falls_back_and_says_so(self, monkeypatch):
        monkeypatch.setattr("algotrader.data._download", lambda *a, **k: None)
        monkeypatch.setattr("algotrader.data._read_cache", lambda *a, **k: None)
        market = load_ohlcv("SPY", "2018-01-01", "2022-01-01")
        assert market.source == "synthetic"
        assert not market.is_real
        assert "unavailable" in market.note

    def test_normalise_handles_yahoo_style_frames(self):
        index = pd.date_range("2020-01-01", periods=5, tz="UTC")
        raw = pd.DataFrame(
            {"Open": 1.0, "High": 2.0, "Low": 0.5, "Adj Close": 1.5, "Volume": 10},
            index=index,
        )
        out = _normalise(raw)
        assert list(out.columns) == ["open", "high", "low", "close", "volume"]
        assert out.index.tz is None

    def test_normalise_rejects_frames_with_no_price(self):
        with pytest.raises(ValueError, match="missing required column"):
            _normalise(pd.DataFrame({"volume": [1, 2, 3]}))

    def test_too_short_a_range_is_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            simulate_ohlcv("SPY", "2020-01-01", "2020-01-10")


class TestVerdict:
    def test_strong_evidence_outranks_weak_evidence(self):
        metrics = {"n_trades": 300, "sharpe": 1.2, "total_return": 0.8, "max_drawdown": -0.2}
        benchmark = {"sharpe": 0.4}
        strong = reality_score(metrics, benchmark, p_value=0.001, dsr=0.99, pbo=0.02,
                               wf_efficiency=0.9, wf_win_rate=1.0, cost_stress_ratio=0.9)
        weak = reality_score(metrics, benchmark, p_value=0.45, dsr=0.10, pbo=0.55,
                             wf_efficiency=-0.2, wf_win_rate=0.2, cost_stress_ratio=0.1)
        assert strong["score"] > 85 > weak["score"]
        assert strong["grade"] == "A" and weak["grade"] == "F"

    def test_score_is_always_inside_the_scale(self):
        for p in (0.0, 0.5, 1.0):
            for dsr in (0.0, 1.0):
                out = reality_score(
                    {"n_trades": 100, "sharpe": 0.5, "total_return": 0.2, "max_drawdown": -0.1},
                    {"sharpe": 0.1}, p_value=p, dsr=dsr, pbo=0.2,
                    wf_efficiency=0.5, cost_stress_ratio=0.5,
                )
                assert 0.0 <= out["score"] <= 100.0

    def test_losing_money_caps_the_score(self):
        out = reality_score(
            {"n_trades": 200, "sharpe": 0.3, "total_return": -0.4, "max_drawdown": -0.6},
            {"sharpe": 0.5}, p_value=0.001, dsr=0.99, pbo=0.01,
            wf_efficiency=1.0, cost_stress_ratio=1.0,
        )
        assert out["score"] <= 50
        assert any("lost money" in f for f in out["flags"])

    def test_too_few_trades_is_flagged_and_capped(self):
        out = reality_score(
            {"n_trades": 3, "sharpe": 2.5, "total_return": 1.0, "max_drawdown": -0.1},
            {"sharpe": 0.3}, p_value=0.001, dsr=0.99, pbo=0.01,
            wf_efficiency=1.0, cost_stress_ratio=1.0,
        )
        assert out["score"] <= 55
        assert any("coin flips" in f for f in out["flags"])

    def test_closet_indexing_is_called_out(self):
        out = reality_score(
            {"n_trades": 50, "sharpe": 0.6, "total_return": 0.5, "max_drawdown": -0.2},
            {"sharpe": 0.6}, benchmark_correlation=0.99,
        )
        assert any("repackaged long position" in f for f in out["flags"])


class TestLabEndToEnd:
    def test_full_pipeline_produces_a_complete_report(self, monkeypatch):
        monkeypatch.setattr("algotrader.lab.load_ohlcv", lambda *a, **k: MarketData(
            "SIM", simulate_ohlcv("SPY", "2016-01-01", "2023-01-01"), "synthetic", "1d", "test"
        ))
        report = run_lab(LabConfig(strategy="sma_cross", n_permutations=25, wf_folds=3, grid_limit=12))

        assert 0.0 <= report.verdict["score"] <= 100.0
        assert report.verdict["grade"] in {"A", "B", "C", "D", "F"}
        assert 0 < report.permutation.p_value <= 1
        assert 0.0 <= report.dsr["dsr"] <= 1.0
        assert report.trials["n"] > 1
        assert len(report.backtest.equity) == len(report.market.df)
        assert report.cost_stress["sharpe_3x"] <= report.cost_stress["sharpe_1x"] + 1e-9

    def test_too_little_history_gives_a_readable_error(self, monkeypatch):
        short = simulate_ohlcv("SPY", "2020-01-01", "2020-06-01")
        monkeypatch.setattr(
            "algotrader.lab.load_ohlcv",
            lambda *a, **k: MarketData("SIM", short, "synthetic", "1d", "test"),
        )
        with pytest.raises(ValueError, match="Widen the date range"):
            run_lab(LabConfig(n_permutations=0, wf_folds=2))

    def test_arena_ranks_every_strategy_and_keeps_the_controls(self, monkeypatch):
        monkeypatch.setattr("algotrader.lab.load_ohlcv", lambda *a, **k: MarketData(
            "SIM", simulate_ohlcv("SPY", "2017-01-01", "2022-01-01"), "synthetic", "1d", "test"
        ))
        table, market, curves = run_arena(LabConfig(), n_permutations=0)

        assert len(table) == len(REGISTRY)
        assert {"buy_and_hold", "coin_flip"} <= set(table["key"])
        assert table["Evidence"].is_monotonic_decreasing
        assert set(curves) == set(REGISTRY)
