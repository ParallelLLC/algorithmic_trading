"""Engine correctness: alignment, costs, and the no-look-ahead guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.engine import bars_to_returns, run_backtest
from algotrader.metrics import compute_metrics, infer_periods_per_year, max_drawdown, sharpe_ratio
from algotrader.types import CostModel


def make_bars(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1e6,
        },
        index=index,
    )


class TestNoLookAhead:
    """The one property the whole project rests on."""

    def test_signal_earns_the_following_bar_not_its_own(self):
        df = make_bars()
        asset_ret = bars_to_returns(df)

        # A target that knows the *next* bar's direction must be perfect...
        clairvoyant = np.sign(asset_ret.shift(-1)).fillna(0.0)
        good = run_backtest(df, clairvoyant, costs=CostModel(0, 0, 0))
        assert (good.returns.iloc[1:-1] >= -1e-12).all(), "perfect foresight should never lose"

        # ...and a target that only knows the *current* bar must not be.
        hindsight = np.sign(asset_ret).fillna(0.0)
        meh = run_backtest(df, hindsight, costs=CostModel(0, 0, 0))
        assert (meh.returns < 0).any(), "same-bar signal must not be risk-free"
        assert good.sharpe > meh.sharpe

    def test_position_is_target_shifted_by_lag(self):
        df = make_bars(200)
        target = pd.Series(np.linspace(-1, 1, len(df)), index=df.index)
        for lag in (1, 2, 5):
            result = run_backtest(df, target, lag=lag)
            expected = target.shift(lag).fillna(0.0)
            pd.testing.assert_series_equal(result.position, expected, check_names=False)

    def test_lag_zero_is_rejected(self):
        df = make_bars(120)
        with pytest.raises(ValueError, match="lag"):
            run_backtest(df, pd.Series(1.0, index=df.index), lag=0)

    def test_future_bars_cannot_change_past_equity(self):
        """Truncating the data must not alter the equity curve before the cut."""
        df = make_bars(400)
        target = pd.Series(np.tile([1.0, -1.0], len(df) // 2), index=df.index)

        full = run_backtest(df, target)
        cut = run_backtest(df.iloc[:250], target.iloc[:250])
        np.testing.assert_allclose(
            full.equity.iloc[:250].to_numpy(), cut.equity.to_numpy(), rtol=1e-12
        )


class TestCosts:
    def test_buy_and_hold_pays_once(self):
        df = make_bars(300)
        target = pd.Series(1.0, index=df.index)
        result = run_backtest(df, target, costs=CostModel(commission_bps=5, slippage_bps=5, short_borrow_bps=0))
        # One entry at 10bps one-way, and no further turnover.
        assert result.costs.sum() == pytest.approx(10 / 1e4, rel=1e-9)
        assert int(result.metrics["n_trades"]) == 1

    def test_flipping_every_bar_costs_more_than_holding(self):
        df = make_bars(300)
        costs = CostModel(commission_bps=5, slippage_bps=5, short_borrow_bps=0)
        hold = run_backtest(df, pd.Series(1.0, index=df.index), costs=costs)
        flip = run_backtest(df, pd.Series(np.tile([1.0, -1.0], 150), index=df.index), costs=costs)
        assert flip.costs.sum() > 100 * hold.costs.sum()

    def test_zero_costs_means_gross_equals_net(self):
        df = make_bars(200)
        target = pd.Series(np.tile([1.0, 0.0], 100), index=df.index)
        result = run_backtest(df, target, costs=CostModel(0, 0, 0))
        pd.testing.assert_series_equal(result.returns, result.gross_returns, check_names=False)

    def test_short_borrow_is_charged_only_on_shorts(self):
        df = make_bars(260)
        costs = CostModel(commission_bps=0, slippage_bps=0, short_borrow_bps=365)
        long_only = run_backtest(df, pd.Series(1.0, index=df.index), costs=costs)
        short_only = run_backtest(df, pd.Series(-1.0, index=df.index), costs=costs)
        assert long_only.costs.sum() == pytest.approx(0.0, abs=1e-12)
        assert short_only.costs.sum() > 0

    def test_higher_costs_never_improve_returns(self):
        df = make_bars(300)
        target = pd.Series(np.tile([1.0, -1.0], 150), index=df.index)
        cheap = run_backtest(df, target, costs=CostModel(1, 1, 0))
        dear = run_backtest(df, target, costs=CostModel(20, 20, 0))
        assert dear.equity.iloc[-1] < cheap.equity.iloc[-1]


class TestConstraints:
    def test_shorts_are_clipped_when_disallowed(self):
        df = make_bars(150)
        target = pd.Series(-1.0, index=df.index)
        result = run_backtest(df, target, allow_short=False)
        assert (result.target >= 0).all()
        assert (result.position >= 0).all()

    def test_leverage_is_clipped(self):
        df = make_bars(150)
        result = run_backtest(df, pd.Series(5.0, index=df.index), max_leverage=1.5)
        assert result.target.max() == pytest.approx(1.5)

    def test_empty_frame_is_rejected(self):
        with pytest.raises(ValueError):
            run_backtest(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), pd.Series(dtype=float))


class TestMetrics:
    def test_sharpe_of_constant_returns_is_zero_not_infinite(self):
        flat = pd.Series([0.001] * 100)
        assert sharpe_ratio(flat, 252) == 0.0

    def test_sharpe_scales_with_annualisation(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.001, 0.01, 5000))
        assert sharpe_ratio(returns, 252) == pytest.approx(sharpe_ratio(returns, 1) * np.sqrt(252))

    def test_max_drawdown_matches_a_hand_worked_example(self):
        equity = pd.Series([100.0, 120.0, 60.0, 90.0])
        assert max_drawdown(equity) == pytest.approx(-0.5)

    def test_buy_and_hold_metrics_match_the_price_series(self):
        df = make_bars(500)
        result = run_backtest(df, pd.Series(1.0, index=df.index), costs=CostModel(0, 0, 0))
        expected = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
        assert result.metrics["total_return"] == pytest.approx(expected, rel=1e-9)

    def test_periodicity_inference(self):
        daily = pd.date_range("2020-01-01", periods=300, freq="B")
        assert 200 <= infer_periods_per_year(daily) <= 300
        hourly = pd.date_range("2020-01-01", periods=300, freq="h")
        assert infer_periods_per_year(hourly) > 1000

    def test_metrics_survive_a_degenerate_series(self):
        empty = pd.Series(dtype=float)
        out = compute_metrics(empty, pd.Series(dtype=float))
        assert "periods_per_year" in out
