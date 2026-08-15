"""Statistical machinery.

These tests matter more than the engine's, because a validation suite that
always says "no edge" is as useless as one that always says "great edge". Each
class below checks both directions: it must reject noise *and* detect signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrader.data import simulate_ohlcv
from algotrader.strategies import get_strategy
from algotrader.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from algotrader.validation.pbo import probability_of_backtest_overfitting
from algotrader.validation.permutation import permutation_test, permute_bars
from algotrader.validation.walkforward import walk_forward


def trending_market(n: int = 2200, phi: float = 0.35, seed: int = 5) -> pd.DataFrame:
    """A market with genuine, exploitable serial correlation."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    noise = rng.normal(0, 0.01, n)
    for i in range(1, n):
        returns[i] = phi * returns[i - 1] + noise[i]
    close = 100 * np.exp(np.cumsum(returns))
    index = pd.date_range("2012-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close * 1.004, "low": close * 0.996, "close": close, "volume": 1e6},
        index=index,
    )


class TestPermutationMechanics:
    def test_shuffling_preserves_the_distribution_of_moves(self):
        df = simulate_ohlcv("SPY", "2018-01-01", "2023-01-01")
        shuffled = permute_bars(df, np.random.default_rng(0))

        assert len(shuffled) == len(df)
        assert shuffled.index.equals(df.index)
        original = np.sort(np.log(df["close"] / df["open"]).to_numpy()[1:])
        permuted = np.sort(np.log(shuffled["close"] / shuffled["open"]).to_numpy()[1:])
        np.testing.assert_allclose(original, permuted, rtol=1e-9)

    def test_shuffling_keeps_bars_internally_valid(self):
        df = simulate_ohlcv("AAPL", "2019-01-01", "2023-01-01")
        shuffled = permute_bars(df, np.random.default_rng(3))
        assert (shuffled["high"] >= shuffled["low"]).all()
        assert (shuffled["high"] >= shuffled["close"]).all()
        assert (shuffled["low"] <= shuffled["close"]).all()
        assert (shuffled["close"] > 0).all()

    def test_shuffling_destroys_serial_correlation(self):
        df = trending_market()
        real = df["close"].pct_change().dropna().autocorr(1)
        shuffled = permute_bars(df, np.random.default_rng(1))["close"].pct_change().dropna().autocorr(1)
        assert real > 0.2
        assert abs(shuffled) < 0.1

    def test_block_mode_retains_some_structure(self):
        df = trending_market()
        blocked = permute_bars(df, np.random.default_rng(2), method="block", block=40)
        assert blocked["close"].pct_change().dropna().autocorr(1) > 0.1

    def test_p_value_can_never_be_zero(self):
        """+1 correction: the observed run is itself a draw from the null."""
        df = trending_market()
        strategy = get_strategy("momentum")
        result = permutation_test(
            df, lambda f: strategy.generate(f, {"lookback": 5}), n_permutations=30, seed=0
        )
        assert result.p_value >= 1 / 31
        assert 0 < result.p_value <= 1


class TestPermutationPower:
    def test_real_edge_is_detected(self):
        df = trending_market()
        strategy = get_strategy("momentum")
        result = permutation_test(
            df, lambda f: strategy.generate(f, {"lookback": 5}), n_permutations=200, seed=1
        )
        assert result.observed > result.null.mean()
        assert result.p_value < 0.05

    def test_random_strategy_on_a_structureless_market_is_not_significant(self):
        df = simulate_ohlcv("SIM", "2010-01-01", "2023-01-01")
        strategy = get_strategy("coin_flip")
        result = permutation_test(
            df, lambda f: strategy.generate(f, {"hold": 5, "seed": 7}), n_permutations=200, seed=2
        )
        assert result.p_value > 0.05


class TestDeflatedSharpe:
    def test_selection_bar_rises_with_the_number_of_trials(self):
        low = expected_max_sharpe(10, 0.01)
        high = expected_max_sharpe(1000, 0.01)
        assert 0 < low < high

    def test_a_single_trial_has_no_selection_bar(self):
        assert expected_max_sharpe(1, 0.01) == 0.0

    def test_more_trials_lowers_the_deflated_sharpe(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0006, 0.01, 2000)
        few = deflated_sharpe_ratio(returns, 1.0, 252, n_trials=2, variance_of_trials=0.01)
        many = deflated_sharpe_ratio(returns, 1.0, 252, n_trials=500, variance_of_trials=0.01)
        assert many["dsr"] < few["dsr"]
        assert many["psr"] == pytest.approx(few["psr"])  # PSR ignores selection

    def test_psr_rises_with_track_record_length(self):
        short = probabilistic_sharpe_ratio(0.05, 100)
        long = probabilistic_sharpe_ratio(0.05, 5000)
        assert 0.5 < short < long < 1.0

    def test_negative_skew_and_fat_tails_are_penalised(self):
        clean = probabilistic_sharpe_ratio(0.06, 1000, skew=0.0, kurtosis=3.0)
        nasty = probabilistic_sharpe_ratio(0.06, 1000, skew=-1.5, kurtosis=12.0)
        assert nasty < clean

    def test_track_record_requirement_is_infinite_below_the_bar(self):
        assert min_track_record_length(0.01, 500, benchmark=0.05) == float("inf")
        assert np.isfinite(min_track_record_length(0.10, 500, benchmark=0.02))


class TestPBO:
    def test_pure_noise_scores_near_one_half(self):
        rng = np.random.default_rng(11)
        matrix = rng.normal(0, 0.01, size=(1200, 30))  # 30 skill-free variants
        result = probability_of_backtest_overfitting(matrix, n_splits=8)
        assert 0.3 < result["pbo"] < 0.7

    def test_a_genuinely_better_variant_is_not_flagged(self):
        rng = np.random.default_rng(12)
        matrix = rng.normal(0, 0.01, size=(1200, 20))
        matrix[:, 3] += 0.004  # column 3 has a persistent, real edge
        result = probability_of_backtest_overfitting(matrix, n_splits=8)
        assert result["pbo"] < 0.15
        assert result["most_selected_index"] == 3
        assert result["selection_stability"] > 0.9

    def test_too_few_variants_returns_nan_not_a_crash(self):
        rng = np.random.default_rng(13)
        result = probability_of_backtest_overfitting(rng.normal(0, 0.01, size=(500, 1)))
        assert np.isnan(result["pbo"])
        assert result["note"]

    def test_odd_split_counts_are_made_even(self):
        rng = np.random.default_rng(14)
        result = probability_of_backtest_overfitting(rng.normal(0, 0.01, (800, 10)), n_splits=7)
        assert result["n_combinations"] > 0


class TestWalkForward:
    def test_a_real_edge_survives_out_of_sample(self):
        result = walk_forward(trending_market(), get_strategy("momentum"), n_folds=4)
        assert result["folds"]
        assert result["mean_oos_sharpe"] > 0
        assert result["efficiency"] > 0.3

    def test_folds_do_not_overlap_train_and_test(self):
        result = walk_forward(trending_market(), get_strategy("sma_cross"), n_folds=4)
        for fold in result["folds"]:
            assert fold["train_end"] <= fold["test_start"]

    def test_short_history_degrades_gracefully(self):
        df = trending_market(n=150)
        result = walk_forward(df, get_strategy("momentum"), n_folds=5)
        assert result["folds"] == []
        assert result["note"]
