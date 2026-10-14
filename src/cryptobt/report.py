"""End-to-end experiment: walk-forward backtests, statistics, tables and figures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .backtest import Costs, drawdown, equal_weight_portfolio, run_backtest, time_in_market, to_equity
from .data import DEFAULT_DATA_DIR, load_prices
from .metrics import summary
from .signals import FAMILIES, buy_and_hold
from .stats import bootstrap_ci, deflated_sharpe, paired_bootstrap_test, probabilistic_sharpe
from .walkforward import walk_forward

ROOT = Path(__file__).resolve().parents[2]

PRETTY = {
    "buy_hold": "Buy & hold",
    "tsmom": "Momentum",
    "tsmom_vt": "Momentum + vol target",
    "ma_cross": "MA crossover",
    "ma_cross_vt": "MA crossover + vol target",
}


@dataclass
class Config:
    assets: tuple[str, ...] = ("btc", "eth")
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    train_days: int = 3 * 365
    test_days: int = 182
    warmup_days: int = 365
    target_vol: float = 0.40
    rebalance_band: float = 0.10
    n_boot: int = 2000
    mean_block: float = 20.0
    seed: int = 7
    data_dir: Path = DEFAULT_DATA_DIR
    out_dir: Path = ROOT / "results"
    end: str | None = None

    @property
    def costs(self) -> Costs:
        return Costs(self.fee_bps, self.slippage_bps)


@dataclass
class AssetRun:
    asset: str
    returns: dict[str, pd.Series] = field(default_factory=dict)  # strategy -> OOS daily net returns
    held: dict[str, pd.Series] = field(default_factory=dict)
    folds: dict[str, pd.DataFrame] = field(default_factory=dict)
    trial_sharpes: list[float] = field(default_factory=list)


def run_asset(close: pd.Series, asset: str, cfg: Config, costs: Costs | None = None, delay: int = 1) -> AssetRun:
    costs = costs or cfg.costs
    run = AssetRun(asset)
    for fam_name, fam in FAMILIES.items():
        for vt in (False, True):
            key = f"{fam_name}_vt" if vt else fam_name
            res = walk_forward(
                close, fam, asset=asset, train_days=cfg.train_days, test_days=cfg.test_days,
                warmup_days=cfg.warmup_days, costs=costs, vol_targeted=vt,
                vt_kwargs={"target_vol": cfg.target_vol, "band": cfg.rebalance_band}, delay=delay,
            )
            run.returns[key] = res.net
            run.held[key] = res.backtest["held"]
            run.folds[key] = res.folds
            run.trial_sharpes.extend(res.trial_sharpes.dropna().tolist())
    oos_start = min(r.index[0] for r in run.returns.values())
    bh = run_backtest(close, buy_and_hold(close), costs, delay).loc[oos_start:]
    run.returns = {"buy_hold": bh["net"], **run.returns}
    run.held = {"buy_hold": bh["held"], **run.held}
    return run


def _fmt_pct(x):
    return f"{x * 100:.1f}%"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def run_all(cfg: Config, verbose: bool = True) -> dict:
    out = Path(cfg.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "folds").mkdir(parents=True, exist_ok=True)
    prices = load_prices(list(cfg.assets), end=cfg.end, data_dir=cfg.data_dir)
    log = print if verbose else (lambda *a, **k: None)

    runs = {a: run_asset(prices[a].dropna(), a, cfg) for a in cfg.assets}
    n_trials = sum(len(f.grid) for f in FAMILIES.values()) * 2  # each family with and without vol target

    # ---- portfolio (equal-weight sleeves) -------------------------------------------------
    strategies = list(next(iter(runs.values())).returns)
    portfolio = {
        s: equal_weight_portfolio(pd.DataFrame({a: runs[a].returns[s] for a in cfg.assets})) for s in strategies
    }
    books = {**{a: runs[a].returns for a in cfg.assets}, "portfolio": portfolio}

    # ---- summary + bootstrap + paired tests -------------------------------------------------
    rows, ci_rows, test_rows = [], [], []
    for book, strat_returns in books.items():
        bench = strat_returns["buy_hold"]
        trials = runs[book].trial_sharpes if book in runs else None
        for s, r in strat_returns.items():
            m = summary(r)
            if book in runs:
                tim = time_in_market(runs[book].held[s])
            else:
                tim = float(np.mean([time_in_market(runs[a].held[s]) for a in cfg.assets]))
            m.update(
                book=book,
                strategy=s,
                time_in_market=tim,
                psr=probabilistic_sharpe(r),
                dsr=deflated_sharpe(r, trials, n_trials) if (trials and s != "buy_hold") else np.nan,
            )
            rows.append(m)
            ci = bootstrap_ci(r, n_boot=cfg.n_boot, mean_block=cfg.mean_block, seed=cfg.seed)
            for stat, v in ci.iterrows():
                ci_rows.append({"book": book, "strategy": s, "stat": stat, **v.to_dict()})
            if s != "buy_hold":
                t = paired_bootstrap_test(r, bench, n_boot=cfg.n_boot, mean_block=cfg.mean_block, seed=cfg.seed)
                test_rows.append({"book": book, "strategy": s, **t})
        log(f"  {book}: done")

    summary_df = pd.DataFrame(rows).set_index(["book", "strategy"])
    ci_df = pd.DataFrame(ci_rows)
    test_df = pd.DataFrame(test_rows).set_index(["book", "strategy"])
    summary_df.to_csv(out / "summary.csv", float_format="%.6f")
    ci_df.to_csv(out / "bootstrap_ci.csv", index=False, float_format="%.6f")
    test_df.to_csv(out / "vs_buy_hold.csv", float_format="%.6f")
    for a, run in runs.items():
        for s, f in run.folds.items():
            f.to_csv(out / "folds" / f"{a}_{s}.csv", index=False, float_format="%.4f")

    # ---- stress tests: double costs, one extra day of execution delay --------------------------
    stress_rows = []
    scenarios = {
        "base": (cfg.costs, 1),
        "2x costs": (Costs(cfg.fee_bps * 2, cfg.slippage_bps * 2), 1),
        "+1 day delay": (cfg.costs, 2),
    }
    for name, (costs, delay) in scenarios.items():
        sruns = runs if name == "base" else {
            a: run_asset(prices[a].dropna(), a, cfg, costs, delay) for a in cfg.assets
        }
        for s in strategies:
            port = equal_weight_portfolio(pd.DataFrame({a: sruns[a].returns[s] for a in cfg.assets}))
            m = summary(port)
            stress_rows.append({"scenario": name, "strategy": s, "cagr": m["cagr"], "sharpe": m["sharpe"],
                                "max_drawdown": m["max_drawdown"]})
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(out / "stress_tests.csv", index=False, float_format="%.6f")
    log("  stress tests: done")

    # ---- figures -----------------------------------------------------------------------------
    best = (
        summary_df.loc["portfolio"].drop(index="buy_hold")["sharpe"].idxmax()
    )
    shown = ["buy_hold", best] + [s for s in ("tsmom", "ma_cross_vt") if s != best][:1]
    port_start = portfolio["buy_hold"].index[0].date()
    port_end = portfolio["buy_hold"].index[-1].date()
    sub = f"Equal-weight BTC + ETH, out-of-sample {port_start} to {port_end}, after {cfg.fee_bps + cfg.slippage_bps:.0f} bps costs per trade"
    plots.equity_chart({PRETTY[s]: to_equity(portfolio[s]) for s in shown}, out / "figures" / "equity.png",
                       "Walk-forward momentum vs buy & hold", sub)
    plots.drawdown_chart({PRETTY[s]: drawdown(to_equity(portfolio[s])) for s in shown},
                         out / "figures" / "drawdown.png", "Drawdowns", sub)
    sh = ci_df[(ci_df.book == "portfolio") & (ci_df.stat == "sharpe")].set_index("strategy")
    sh = sh.loc[strategies]
    sh["is_benchmark"] = sh.index == "buy_hold"
    sh.index = [PRETTY[s] for s in sh.index]
    plots.sharpe_ci_chart(sh, out / "figures" / "sharpe_ci.png", "How sure are we about the Sharpe ratio?",
                          f"Stationary block bootstrap, {cfg.n_boot:,} resamples, mean block {cfg.mean_block:.0f} days")

    # ---- markdown report ---------------------------------------------------------------------
    write_markdown(cfg, summary_df, test_df, stress_df, runs, out / "REPORT.md", n_trials)
    log(f"Results written to {out}")
    return {"summary": summary_df, "tests": test_df, "stress": stress_df, "ci": ci_df, "best": best}


def write_markdown(cfg, summary_df, test_df, stress_df, runs, path: Path, n_trials: int) -> None:
    def table_for(book):
        df = summary_df.loc[book].copy()
        t = test_df.loc[book] if book in test_df.index.get_level_values(0) else None
        view = pd.DataFrame(index=[PRETTY[s] for s in df.index])
        view.index.name = "Strategy"
        view["CAGR"] = [_fmt_pct(x) for x in df.cagr]
        view["Volatility"] = [_fmt_pct(x) for x in df.ann_vol]
        view["Sharpe"] = [f"{x:.2f}" for x in df.sharpe]
        view["Max drawdown"] = [_fmt_pct(x) for x in df.max_drawdown]
        view["Calmar"] = [f"{x:.2f}" for x in df.calmar]
        view["Time in market"] = ["-" if np.isnan(x) else _fmt_pct(x) for x in df.time_in_market]
        view["Deflated Sharpe"] = ["-" if np.isnan(x) else f"{x:.2f}" for x in df.dsr]
        pv = []
        for s in df.index:
            pv.append("-" if t is None or s not in t.index else f"{t.loc[s, 'p_value']:.2f}")
        view["p (Sharpe > B&H)"] = pv
        return _md_table(view)

    lines = [
        "# Results",
        "",
        "_Generated by `python -m cryptobt run`. Every number below is out-of-sample: parameters were",
        f"chosen on the previous {cfg.train_days // 365} years and then traded unchanged for the next {cfg.test_days} days._",
        "",
        f"Costs: {cfg.fee_bps:.0f} bps fee + {cfg.slippage_bps:.0f} bps slippage per unit of turnover. "
        f"Strategy variants tried per asset: {n_trials}. Bootstrap: {cfg.n_boot:,} stationary-block resamples.",
        "",
    ]
    for book in ["portfolio", *cfg.assets]:
        df = summary_df.loc[book]
        title = "Portfolio (equal-weight BTC + ETH)" if book == "portfolio" else book.upper()
        lines += [f"## {title}", "", f"Out-of-sample period: {df.start.iloc[0]} to {df.end.iloc[0]}", "", table_for(book), ""]

    lines += ["## Stress tests (portfolio Sharpe)", ""]
    order = list(dict.fromkeys(stress_df.strategy))
    pivot = stress_df.pivot(index="strategy", columns="scenario", values="sharpe").loc[order, ["base", "2x costs", "+1 day delay"]]
    pivot.index = [PRETTY[s] for s in pivot.index]
    pivot.index.name = "Strategy"
    lines += [_md_table(pivot.map(lambda x: f"{x:.2f}")), ""]

    lines += ["## Parameters chosen in each fold", ""]
    for a, run in runs.items():
        for s in ("tsmom", "ma_cross"):
            f = run.folds[s]
            chosen = ", ".join(f"{r.test_start:%Y-%m}: {r.params}" for r in f.itertuples())
            lines.append(f"- **{a.upper()} {PRETTY[s]}** - {chosen}")
    lines.append("")
    path.write_text("\n".join(lines))
