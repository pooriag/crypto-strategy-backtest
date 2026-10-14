"""Command line interface.

    python -m cryptobt fetch --assets btc eth      # refresh data/ from Coin Metrics
    python -m cryptobt run                         # full experiment -> results/
    python -m cryptobt run --fee-bps 5 --n-boot 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_DATA_DIR, clean_prices, fetch_coinmetrics, save_snapshot
from .report import Config, run_all


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cryptobt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download and clean daily prices")
    f.add_argument("--assets", nargs="+", default=["btc", "eth"])
    f.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    r = sub.add_parser("run", help="run the full walk-forward experiment")
    r.add_argument("--assets", nargs="+", default=["btc", "eth"])
    r.add_argument("--fee-bps", type=float, default=10.0)
    r.add_argument("--slippage-bps", type=float, default=5.0)
    r.add_argument("--train-days", type=int, default=3 * 365)
    r.add_argument("--test-days", type=int, default=182)
    r.add_argument("--target-vol", type=float, default=0.40)
    r.add_argument("--rebalance-band", type=float, default=0.10, help="no-trade band for vol-targeted variants")
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--end", default=None, help="last date to use (YYYY-MM-DD)")
    r.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    r.add_argument("--out-dir", type=Path, default=Config.out_dir)

    a = p.parse_args(argv)
    if a.cmd == "fetch":
        for asset in a.assets:
            s, rep = clean_prices(fetch_coinmetrics(asset), asset)
            path = save_snapshot(asset, s, a.data_dir)
            print(f"{rep}  ->  {path}")
    else:
        cfg = Config(
            assets=tuple(a.assets), fee_bps=a.fee_bps, slippage_bps=a.slippage_bps, train_days=a.train_days,
            test_days=a.test_days, target_vol=a.target_vol, rebalance_band=a.rebalance_band, n_boot=a.n_boot, end=a.end,
            data_dir=a.data_dir, out_dir=a.out_dir,
        )
        res = run_all(cfg)
        cols = ["cagr", "sharpe", "max_drawdown", "time_in_market"]
        print(res["summary"].loc["portfolio"][cols].round(3).to_string())


if __name__ == "__main__":
    main()
