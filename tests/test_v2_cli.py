"""CLI surface — argument handling and that each command actually completes."""

from __future__ import annotations

import json

import pytest

from algotrader.cli import _parse_params, main


class TestArgumentParsing:
    def test_params_are_parsed_into_floats(self):
        assert _parse_params(["fast=10", "slow=50.5"]) == {"fast": 10.0, "slow": 50.5}

    def test_params_without_an_equals_sign_are_rejected(self):
        with pytest.raises(SystemExit, match="name=value"):
            _parse_params(["fast10"])

    def test_no_params_is_an_empty_dict(self):
        assert _parse_params(None) == {}

    def test_an_unknown_strategy_is_rejected_at_parse_time(self):
        with pytest.raises(SystemExit):
            main(["lab", "--strategy", "nope"])

    def test_a_missing_subcommand_is_rejected(self):
        with pytest.raises(SystemExit):
            main([])


class TestCommands:
    """All runs force the simulator so the suite never touches the network."""

    def test_lab_prints_a_report(self, capsys):
        code = main([
            "lab", "--symbol", "SPY", "--source", "synthetic", "--strategy", "sma_cross",
            "--start", "2018-01-01", "--end", "2023-01-01",
            "--permutations", "10", "--folds", "2", "--quiet",
        ])
        out = capsys.readouterr().out
        assert code == 0
        assert "REALITY SCORE" in out
        assert "Deflated Sharpe" in out

    def test_lab_json_output_is_machine_readable(self, capsys):
        code = main([
            "lab", "--source", "synthetic", "--start", "2018-01-01", "--end", "2023-01-01",
            "--permutations", "10", "--folds", "2", "--quiet", "--json",
        ])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert 0.0 <= payload["verdict"]["score"] <= 100.0
        assert payload["verdict"]["grade"] in {"A", "B", "C", "D", "F"}
        assert payload["source"] == "synthetic"

    def test_lab_honours_param_overrides(self, capsys):
        main([
            "lab", "--source", "synthetic", "--start", "2018-01-01", "--end", "2023-01-01",
            "--strategy", "sma_cross", "--param", "fast=5", "--param", "slow=60",
            "--permutations", "0", "--folds", "2", "--quiet", "--json",
        ])
        payload = json.loads(capsys.readouterr().out)
        assert payload["params"] == {"fast": 5, "slow": 60}
        assert payload["p_value"] is None  # permutations disabled

    def test_arena_ranks_the_zoo(self, capsys):
        code = main([
            "arena", "--source", "synthetic", "--start", "2018-01-01", "--end", "2023-01-01",
            "--permutations", "0", "--quiet",
        ])
        out = capsys.readouterr().out
        assert code == 0
        assert "Buy & Hold" in out and "Coin Flip" in out
        assert "Ranked by evidence" in out

    def test_strategies_lists_the_registry(self, capsys):
        assert main(["strategies"]) == 0
        out = capsys.readouterr().out
        assert "sma_cross" in out and "donchian_breakout" in out

    def test_errors_are_reported_not_raised(self, capsys):
        code = main([
            "lab", "--source", "synthetic",
            "--start", "2022-01-01", "--end", "2022-02-01",  # far too short
            "--permutations", "0", "--quiet",
        ])
        assert code == 1
        assert "error:" in capsys.readouterr().err
