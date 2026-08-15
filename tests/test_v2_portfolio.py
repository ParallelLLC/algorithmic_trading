"""Multi-asset engine, panel, cross-sectional strategies and their null."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.attribution import build_style_factors, factor_attribution
from algotrader.cross_sectional import XS_REGISTRY, get_xs_strategy, scores_to_weights
from algotrader.data import simulate_ohlcv
from algotrader.engine import run_backtest
from algotrader.panel import Panel, load_panel
from algotrader.portfolio import (
    normalise_weights,
    rebalance_schedule,
    run_portfolio_backtest,
)
from algotrader.types import CostModel
from algotrader.validation.cross_permutation import (
    cross_sectional_permutation_test,
    permute_within_dates,
)

XS_KEYS = sorted(XS_REGISTRY)


def make_panel(n_assets=8, n=1200, drift_sd=0.0, seed=1, common_vol=0.008, idio=0.012) -> Panel:
    """A synthetic universe. ``drift_sd`` controls genuine cross-sectional structure."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2016-01-01", periods=n, freq="B")
    drift = rng.normal(0, drift_sd, n_assets)
    returns = (
        rng.normal(0.0002, common_vol, n)[:, None]
        + rng.normal(0, idio, (n, n_assets))
        + drift[None, :]
    )
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    columns = [f"A{i}" for i in range(n_assets)]

    def frame(values):
        return pd.DataFrame(values, index=index, columns=columns)

    return Panel(
        fields={
            "open": frame(close),
            "high": frame(close * 1.004),
            "low": frame(close * 0.996),
            "close": frame(close),
            "volume": frame(np.full_like(close, 1e6)),
        },
        sources={c: "synthetic" for c in columns},
    )


@pytest.fixture(scope="module")
def panel() -> Panel:
    return make_panel()


class TestPanel:
    def test_fields_are_aligned(self, panel):
        assert panel.shape == (1200, 8)
        for name in ("open", "high", "low", "close", "volume"):
            assert panel.fields[name].index.equals(panel.index)
            assert list(panel.fields[name].columns) == panel.symbols

    def test_misaligned_fields_are_rejected(self, panel):
        broken = dict(panel.fields)
        broken["high"] = broken["high"].iloc[:-5]
        with pytest.raises(ValueError, match="not aligned"):
            Panel(fields=broken)

    def test_missing_field_is_rejected(self, panel):
        with pytest.raises(ValueError, match="missing field"):
            Panel(fields={"close": panel.close})

    def test_missing_data_is_not_forward_filled(self):
        """Filling a gap invents liquidity that never existed."""
        df = simulate_ohlcv("AAPL", "2018-01-01", "2022-01-01")
        gapped = df.drop(df.index[100:140])
        built = Panel.from_frames({"AAPL": gapped, "MSFT": df})
        assert built.close["AAPL"].isna().sum() == 40

    def test_tradable_requires_two_consecutive_prices(self, panel):
        assert not panel.tradable().iloc[0].any()
        assert panel.tradable().iloc[1:].all().all()

    def test_returns_are_masked_where_untradable(self, panel):
        assert panel.returns().iloc[0].isna().all()

    def test_delisting_is_detected(self):
        df = simulate_ohlcv("AAPL", "2016-01-01", "2022-01-01")
        dead = df.iloc[: len(df) // 2]
        built = Panel.from_frames({"ALIVE": df, "DEAD": dead})
        report = built.survivorship()
        assert report.n_delisted == 1
        assert "DEAD" in report.delisted_symbols
        assert not report.biased

    def test_all_survivors_is_flagged_as_biased(self, panel):
        report = panel.survivorship()
        assert report.survival_rate == 1.0
        assert report.biased
        assert "upper bound" in report.note

    def test_load_panel_skips_symbols_without_history(self):
        built = load_panel(["SPY", "AAPL"], "2018-01-01", "2022-01-01", source="synthetic")
        assert set(built.symbols) == {"SPY", "AAPL"}
        assert all(s == "synthetic" for s in built.sources.values())

    def test_select_and_slice(self, panel):
        subset = panel.select(["A0", "A1"])
        assert subset.symbols == ["A0", "A1"]
        sliced = panel.slice(panel.index[10], panel.index[50])
        assert len(sliced) == 41


class TestPortfolioEngine:
    def test_matches_single_asset_engine_exactly(self):
        """One column through the matrix engine must equal the fast path."""
        df = simulate_ohlcv("SPY", "2018-01-01", "2023-01-01")
        single = Panel.from_frames({"SPY": df})
        costs = CostModel(1, 2, 50)
        n = len(df)
        cases = {
            "buy_and_hold": np.ones(n),
            "flip": np.tile([1.0, -1.0], n // 2 + 1)[:n],
            "long_flat": np.tile([1.0, 0.0], n // 2 + 1)[:n],
            "fractional": np.full(n, 0.5),
        }
        for name, target in cases.items():
            portfolio = run_portfolio_backtest(
                single, pd.DataFrame({"SPY": target}, index=df.index), costs=costs
            )
            direct = run_backtest(df, pd.Series(target, index=df.index), costs=costs)
            np.testing.assert_allclose(
                portfolio.returns.to_numpy(), direct.returns.to_numpy(),
                atol=1e-12, err_msg=f"mismatch for {name}",
            )

    def test_holding_still_costs_nothing_but_drift_does(self, panel):
        """A full-notional long needs no rebalancing; a short does."""
        costs = CostModel(10, 10, 0)
        long_only = pd.DataFrame(1.0 / len(panel.symbols), index=panel.index, columns=panel.symbols)
        result = run_portfolio_backtest(panel, long_only, costs=costs, gross_leverage=1.0)
        # Equal-weight across many names still drifts apart, so turnover > 0...
        assert result.metrics["turnover_ann"] > 0
        # ...but a single full-notional position does not.
        one = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
        one["A0"] = 1.0
        assert run_portfolio_backtest(panel, one, costs=costs).costs.sum() == pytest.approx(
            20 / 1e4, rel=1e-6
        )

    def test_rebalancing_less_often_lowers_turnover(self, panel):
        weights = get_xs_strategy("equal_weight").generate(panel)
        turnovers = []
        for frequency in ("D", "M", "Q"):
            result = run_portfolio_backtest(
                panel, weights, costs=CostModel(1, 2, 0),
                rebalance_on=rebalance_schedule(panel.index, frequency),
            )
            turnovers.append(result.metrics["turnover_ann"])
        assert turnovers[0] > turnovers[1] > turnovers[2]

    def test_gross_leverage_is_capped(self, panel):
        weights = pd.DataFrame(1.0, index=panel.index, columns=panel.symbols)
        result = run_portfolio_backtest(panel, weights, gross_leverage=1.0)
        assert result.weights.abs().sum(axis=1).max() <= 1.0 + 1e-9

    def test_leverage_is_scaled_down_never_up(self, panel):
        small = pd.DataFrame(0.01, index=panel.index, columns=panel.symbols)
        result = run_portfolio_backtest(panel, small, gross_leverage=1.0)
        assert result.weights.abs().sum(axis=1).max() < 0.2

    def test_per_name_cap_is_applied(self, panel):
        weights = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
        weights["A0"] = 1.0
        result = run_portfolio_backtest(panel, weights, max_weight=0.1)
        assert result.weights.abs().max().max() <= 0.1 + 1e-9

    def test_shorts_are_blocked_when_disallowed(self, panel):
        weights = pd.DataFrame(-0.1, index=panel.index, columns=panel.symbols)
        result = run_portfolio_backtest(panel, weights, allow_short=False)
        assert (result.weights >= 0).all().all()

    def test_delisted_names_cannot_be_held(self):
        df = simulate_ohlcv("AAPL", "2016-01-01", "2022-01-01")
        built = Panel.from_frames({"ALIVE": df, "DEAD": df.iloc[: len(df) // 2]})
        weights = pd.DataFrame(0.5, index=built.index, columns=built.symbols)
        result = run_portfolio_backtest(built, weights)
        after_death = built.close["DEAD"].last_valid_index()
        assert result.held.loc[result.held.index > after_death, "DEAD"].abs().max() == 0.0

    def test_lag_zero_is_rejected(self, panel):
        with pytest.raises(ValueError, match="lag"):
            run_portfolio_backtest(panel, pd.DataFrame(0.1, index=panel.index, columns=panel.symbols), lag=0)

    def test_future_bars_cannot_change_past_equity(self, panel):
        weights = get_xs_strategy("xs_momentum").generate(panel)
        full = run_portfolio_backtest(panel, weights)
        cut = run_portfolio_backtest(
            panel.slice(panel.index[0], panel.index[799]), weights.iloc[:800]
        )
        np.testing.assert_allclose(
            full.equity.iloc[:800].to_numpy(), cut.equity.to_numpy(), rtol=1e-10
        )

    def test_attribution_sums_to_gross_return(self, panel):
        weights = get_xs_strategy("equal_weight").generate(panel)
        result = run_portfolio_backtest(panel, weights, costs=CostModel(0, 0, 0))
        assert result.attribution().sum() == pytest.approx(result.gross_returns.sum(), rel=1e-9)

    def test_portfolio_metrics_are_reported(self, panel):
        result = run_portfolio_backtest(panel, get_xs_strategy("xs_momentum").generate(panel))
        for key in ("gross_exposure", "net_exposure", "avg_positions", "concentration_hhi"):
            assert key in result.metrics


class TestCrossSectionalStrategies:
    @pytest.mark.parametrize("key", XS_KEYS)
    def test_weights_are_well_formed(self, panel, key):
        weights = get_xs_strategy(key).generate(panel)
        assert weights.shape == panel.shape
        assert weights.notna().all().all()
        assert weights.abs().sum(axis=1).max() <= 1.0 + 1e-9

    @pytest.mark.parametrize("key", XS_KEYS)
    def test_weights_are_causal(self, panel, key):
        cut = 700
        full = get_xs_strategy(key).generate(panel).iloc[:cut]
        partial = get_xs_strategy(key).generate(panel.slice(panel.index[0], panel.index[cut - 1]))
        pd.testing.assert_frame_equal(full, partial, rtol=1e-9)

    def test_long_short_books_are_dollar_neutral(self, panel):
        weights = get_xs_strategy("xs_momentum").generate(panel)
        active = weights[weights.abs().sum(axis=1) > 1e-9]
        assert active.sum(axis=1).abs().max() < 1e-9

    def test_long_only_uses_the_whole_book(self):
        scores = pd.DataFrame(np.arange(40).reshape(4, 10).astype(float))
        weights = scores_to_weights(scores, long_frac=0.3, long_only=True)
        assert weights.sum(axis=1).round(9).eq(1.0).all()
        assert (weights >= 0).all().all()

    def test_thin_cross_sections_are_skipped(self):
        scores = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 3)))
        assert scores_to_weights(scores).abs().sum(axis=1).max() == 0.0

    def test_equal_weight_holds_everything(self, panel):
        weights = get_xs_strategy("equal_weight").generate(panel)
        assert (weights.iloc[-1] > 0).all()

    def test_unknown_strategy_names_alternatives(self):
        with pytest.raises(KeyError, match="Available"):
            get_xs_strategy("nope")


class TestCrossSectionalNull:
    def test_permutation_preserves_the_shape_of_the_book(self, panel):
        weights = get_xs_strategy("xs_momentum").generate(panel)
        investable = panel.close.notna()
        shuffled = permute_within_dates(weights, investable, np.random.default_rng(0))

        np.testing.assert_allclose(
            np.sort(weights.to_numpy(), axis=1), np.sort(shuffled.to_numpy(), axis=1), atol=1e-12
        )
        np.testing.assert_allclose(
            weights.abs().sum(axis=1).to_numpy(), shuffled.abs().sum(axis=1).to_numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            weights.sum(axis=1).to_numpy(), shuffled.sum(axis=1).to_numpy(), atol=1e-12
        )

    def test_permutation_actually_reassigns(self, panel):
        weights = get_xs_strategy("xs_momentum").generate(panel)
        shuffled = permute_within_dates(weights, panel.close.notna(), np.random.default_rng(0))
        assert not np.allclose(weights.to_numpy(), shuffled.to_numpy())

    def test_non_investable_names_stay_empty(self):
        df = simulate_ohlcv("AAPL", "2016-01-01", "2022-01-01")
        built = Panel.from_frames({"ALIVE": df, "DEAD": df.iloc[: len(df) // 2]})
        weights = pd.DataFrame(0.5, index=built.index, columns=built.symbols)
        shuffled = permute_within_dates(weights, built.close.notna(), np.random.default_rng(1))
        after_death = built.close["DEAD"].last_valid_index()
        assert shuffled.loc[shuffled.index > after_death, "DEAD"].abs().max() == 0.0

    def test_real_cross_sectional_skill_is_detected(self):
        strong = make_panel(drift_sd=0.003, seed=11)
        weights = get_xs_strategy("xs_momentum").generate(strong)
        result = cross_sectional_permutation_test(
            strong, weights, n_permutations=100,
            rebalance_on=rebalance_schedule(strong.index, "M"), seed=2,
        )
        assert result.observed > result.null_mean
        assert result.p_value < 0.05

    def test_no_structure_is_not_significant(self):
        flat = make_panel(drift_sd=0.0, seed=12)
        weights = get_xs_strategy("xs_momentum").generate(flat)
        result = cross_sectional_permutation_test(
            flat, weights, n_permutations=100,
            rebalance_on=rebalance_schedule(flat.index, "M"), seed=4,
        )
        assert result.p_value > 0.05

    def test_p_value_can_never_be_zero(self, panel):
        weights = get_xs_strategy("xs_momentum").generate(panel)
        result = cross_sectional_permutation_test(panel, weights, n_permutations=20, seed=0)
        assert result.p_value >= 1 / 21


class TestAttribution:
    def test_equal_weight_is_explained_by_the_market(self, panel):
        factors = build_style_factors(panel)
        weights = get_xs_strategy("equal_weight").generate(panel)
        result = run_portfolio_backtest(panel, weights)
        report = factor_attribution(result.returns, factors)

        assert report["available"]
        assert report["r_squared"] > 0.85
        assert report["dominant_factor"] == "market"
        assert not report["alpha_significant"]
        assert "more cheaply" in report["note"]

    def test_a_factor_regressed_on_itself_has_no_alpha(self, panel):
        factors = build_style_factors(panel)
        report = factor_attribution(factors["market"], factors)
        assert abs(report["alpha_annual"]) < 0.05
        assert report["r_squared"] > 0.99

    def test_pure_noise_has_no_alpha_and_no_fit(self, panel):
        rng = np.random.default_rng(5)
        noise = pd.Series(rng.normal(0, 0.01, len(panel)), index=panel.index)
        report = factor_attribution(noise, build_style_factors(panel))
        assert not report["alpha_significant"]
        assert report["r_squared"] < 0.2

    def test_short_series_degrade_gracefully(self, panel):
        factors = build_style_factors(panel)
        report = factor_attribution(factors["market"].iloc[:20], factors)
        assert not report["available"]
        assert report["note"]


class TestPortfolioLab:
    def test_full_pipeline(self):
        from algotrader.portfolio_lab import PortfolioLabConfig, run_portfolio_lab

        report = run_portfolio_lab(
            PortfolioLabConfig(
                symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GLD"],
                start="2017-01-01", end="2023-01-01", source="synthetic",
                strategy="xs_momentum", n_permutations=20, wf_folds=2, grid_limit=6,
            )
        )
        assert 0.0 <= report.verdict["score"] <= 100.0
        assert report.verdict["grade"] in {"A", "B", "C", "D", "F"}
        assert 0 < report.permutation.p_value <= 1
        assert report.attribution["available"]
        assert report.survivorship.n_symbols == 6
        assert report.cost_stress["sharpe_3x"] <= report.cost_stress["sharpe_1x"] + 1e-9

    def test_survivorship_bias_reaches_the_verdict(self):
        from algotrader.portfolio_lab import PortfolioLabConfig, run_portfolio_lab

        report = run_portfolio_lab(
            PortfolioLabConfig(
                symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GLD"],
                start="2015-01-01", end="2023-01-01", source="synthetic",
                strategy="equal_weight", n_permutations=0, wf_folds=2, grid_limit=3,
            )
        )
        assert report.survivorship.biased
        assert any("still trading" in f for f in report.verdict["flags"])
        assert report.verdict["score"] <= 60
