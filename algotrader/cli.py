"""Command-line interface.

    python -m algotrader.cli lab --symbol SPY --strategy sma_cross
    python -m algotrader.cli arena --symbol BTC-USD --start 2018-01-01
    python -m algotrader.cli strategies
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from . import __version__
from .lab import LabConfig, run_arena, run_lab
from .strategies import REGISTRY, get_strategy


def _parse_params(pairs: List[str] | None) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param expects name=value, got '{pair}'")
        name, _, value = pair.partition("=")
        params[name.strip()] = float(value)
    return params


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument(
        "--source", default="auto", choices=["auto", "cache", "synthetic"],
        help="'auto' downloads and falls back offline; 'synthetic' forces the simulator.",
    )
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--no-short", action="store_true", help="Long/flat only.")


def _config_from(args: argparse.Namespace, **overrides) -> LabConfig:
    return LabConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval,
        source=args.source,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        allow_short=not args.no_short,
        **overrides,
    )


def _cmd_lab(args: argparse.Namespace) -> int:
    cfg = _config_from(
        args,
        strategy=args.strategy,
        params=_parse_params(args.param),
        n_permutations=args.permutations,
        permutation_method=args.null,
        wf_folds=args.folds,
    )
    progress = None if args.quiet else (lambda f, m: print(f"  [{f:5.0%}] {m}", file=sys.stderr))
    report = run_lab(cfg, progress=progress)

    if args.json:
        payload = {
            "symbol": report.market.symbol,
            "source": report.market.source,
            "strategy": report.strategy.key,
            "params": report.params,
            "metrics": report.backtest.metrics,
            "benchmark_metrics": report.backtest.benchmark_metrics,
            "p_value": report.permutation.p_value if report.permutation else None,
            "deflated_sharpe": report.dsr.get("dsr"),
            "pbo": report.pbo.get("pbo"),
            "walkforward_efficiency": report.walkforward.get("efficiency"),
            "cost_stress": report.cost_stress,
            "verdict": {k: v for k, v in report.verdict.items()},
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    v, m, b = report.verdict, report.backtest.metrics, report.backtest.benchmark_metrics
    bar = "=" * 66
    print(f"\n{bar}")
    print(f"  {report.strategy.name} on {report.market.symbol}   [{report.market.source} data]")
    print(f"  {report.market.start.date()} to {report.market.end.date()}  ·  {len(report.market.df):,} bars")
    print(bar)
    print(f"  REALITY SCORE   {v['score']:.1f} / 100      GRADE  {v['grade']}")
    print(f"  {v['headline']}")
    print(bar)
    print(f"  Total return      {m['total_return']:>9.1%}    buy & hold {b['total_return']:>8.1%}")
    print(f"  CAGR              {m['cagr']:>9.1%}    buy & hold {b['cagr']:>8.1%}")
    print(f"  Sharpe            {m['sharpe']:>9.2f}    buy & hold {b['sharpe']:>8.2f}")
    print(f"  Max drawdown      {m['max_drawdown']:>9.1%}")
    print(f"  Trades            {int(m.get('n_trades', 0)):>9,}")
    print(bar)
    if report.permutation:
        print(f"  Permutation p     {report.permutation.p_value:>9.3f}    ({report.permutation.n_permutations} shuffled markets)")
    print(f"  Deflated Sharpe   {report.dsr.get('dsr', 0):>9.2f}    (after {report.trials.get('n', 1)} variants)")
    pbo = report.pbo.get("pbo")
    print(f"  Overfit prob.     {pbo:>9.2f}" if pbo == pbo else "  Overfit prob.           n/a")
    print(f"  Walk-forward eff. {report.walkforward.get('efficiency', 0):>9.2f}")
    print(f"  Sharpe at 3x cost {report.cost_stress.get('sharpe_3x', 0):>9.2f}")
    print(bar)
    for flag in v["flags"]:
        print(f"  ! {flag}")
    if v["flags"]:
        print(bar)
    print(f"  {v['verdict']}\n")
    return 0


def _cmd_arena(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    progress = None if args.quiet else (lambda f, m: print(f"  [{f:5.0%}] {m}", file=sys.stderr))
    table, market, _ = run_arena(cfg, n_permutations=args.permutations, progress=progress)

    if args.json:
        print(table.to_json(orient="records", indent=2))
        return 0

    print(f"\n  {market.symbol}  [{market.source} data]  "
          f"{market.start.date()} to {market.end.date()}\n")
    display = table.drop(columns=["key"]).copy()
    for col in ("Return", "CAGR", "MaxDD"):
        display[col] = display[col].map("{:.1%}".format)
    for col in ("Sharpe", "DSR", "Evidence"):
        display[col] = display[col].map("{:.2f}".format)
    display["p-value"] = display["p-value"].map(lambda v: "—" if v != v else f"{v:.3f}")
    print(display.to_string(index=False))
    print("\n  Ranked by evidence = (1 - p) x deflated Sharpe, not by return.\n")
    return 0


def _cmd_strategies(args: argparse.Namespace) -> int:
    for key, strategy in REGISTRY.items():
        params = ", ".join(f"{p.name}={p.default:g}" for p in strategy.params) or "no parameters"
        print(f"  {key:<22} {strategy.name:<26} [{strategy.family}]")
        print(f"  {'':<22} {strategy.description}")
        print(f"  {'':<22} defaults: {params}\n")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="algotrader",
        description="Backtest a trading rule, then try to prove the result was luck.",
    )
    parser.add_argument("--version", action="version", version=f"algotrader {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    lab = sub.add_parser("lab", help="Full reality check for one strategy.")
    _add_common(lab)
    lab.add_argument("--strategy", default="sma_cross", choices=sorted(REGISTRY))
    lab.add_argument("--param", action="append", metavar="NAME=VALUE",
                     help="Override a strategy parameter. Repeatable.")
    lab.add_argument("--permutations", type=int, default=250)
    lab.add_argument("--null", default="permute", choices=["permute", "block"])
    lab.add_argument("--folds", type=int, default=5)
    lab.add_argument("--json", action="store_true")
    lab.add_argument("--quiet", "-q", action="store_true")
    lab.set_defaults(func=_cmd_lab)

    arena = sub.add_parser("arena", help="Race every strategy on one market.")
    _add_common(arena)
    arena.add_argument("--permutations", type=int, default=120)
    arena.add_argument("--json", action="store_true")
    arena.add_argument("--quiet", "-q", action="store_true")
    arena.set_defaults(func=_cmd_arena)

    listing = sub.add_parser("strategies", help="List the strategy zoo.")
    listing.set_defaults(func=_cmd_strategies)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
