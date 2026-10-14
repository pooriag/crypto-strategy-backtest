"""Loading and cleaning daily price data.

The default source is the free Coin Metrics community archive hosted on GitHub
(https://github.com/coinmetrics/data), which publishes one CSV per asset with a
daily USD reference price (``PriceUSD``). It needs no API key and is reachable
from most networks, which is why it is preferred over exchange APIs here.

A cleaned snapshot (``date,close``) is stored in ``data/<asset>.csv`` so the
whole project runs offline; ``cryptobt fetch`` refreshes it.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COINMETRICS_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{asset}.csv"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Dates before which prices are too thin/illiquid to trade realistically.
DEFAULT_START = {"btc": "2014-01-01", "eth": "2016-01-01"}


@dataclass
class CleaningReport:
    asset: str
    rows_in: int
    rows_out: int
    non_positive_dropped: int
    gaps_filled: int
    first_date: str
    last_date: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.asset}: {self.rows_out} rows ({self.first_date} -> {self.last_date}), "
            f"dropped {self.non_positive_dropped} bad prices, filled {self.gaps_filled} gap days"
        )


def parse_coinmetrics_csv(raw: str | bytes) -> pd.Series:
    """Parse a Coin Metrics community CSV into a daily close-price Series."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), usecols=["time", "PriceUSD"])
    s = df.set_index(pd.to_datetime(df["time"]))["PriceUSD"].astype(float)
    s.index.name = "date"
    s.name = "close"
    return s


def clean_prices(s: pd.Series, asset: str = "", max_gap: int = 3) -> tuple[pd.Series, CleaningReport]:
    """Drop missing/non-positive prices, put on a daily calendar and forward-fill short gaps.

    Crypto trades every day, so any missing calendar day is a data gap. Gaps of up
    to ``max_gap`` days are forward-filled (a flat price = zero return, which is
    conservative); longer gaps are left as NaN and trimmed from the ends.
    """
    rows_in = len(s)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    bad = s.notna() & (s <= 0)
    s = s[~bad].dropna()
    if s.empty:
        raise ValueError(f"no valid prices for {asset!r}")
    full = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="D"))
    full.index.name = "date"
    n_missing = int(full.isna().sum())
    full = full.ffill(limit=max_gap)
    gaps_filled = n_missing - int(full.isna().sum())
    full = full.dropna()
    full.name = "close"
    report = CleaningReport(
        asset=asset,
        rows_in=rows_in,
        rows_out=len(full),
        non_positive_dropped=int(bad.sum()),
        gaps_filled=gaps_filled,
        first_date=str(full.index[0].date()),
        last_date=str(full.index[-1].date()),
    )
    return full, report


def fetch_coinmetrics(asset: str, timeout: int = 60) -> pd.Series:
    """Download and parse the Coin Metrics community CSV for one asset."""
    url = COINMETRICS_URL.format(asset=asset.lower())
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        raw = resp.read()
    return parse_coinmetrics_csv(raw)


def save_snapshot(asset: str, s: pd.Series, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{asset.lower()}.csv"
    s.rename("close").to_frame().to_csv(path, float_format="%.6f")
    return path


def load_snapshot(asset: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.Series:
    path = Path(data_dir) / f"{asset.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m cryptobt fetch --assets {asset}` first")
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["close"].astype(float)


def load_prices(
    assets: list[str],
    start: str | None = None,
    end: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load cleaned close prices for several assets as one DataFrame (columns = assets).

    Each asset keeps its own history; ``start`` overrides the per-asset default start.
    Rows where an asset has not started yet are NaN.
    """
    cols = {}
    for a in assets:
        s = load_snapshot(a, data_dir)
        a_start = start or DEFAULT_START.get(a.lower())
        if a_start:
            s = s[s.index >= pd.Timestamp(a_start)]
        if end:
            s = s[s.index <= pd.Timestamp(end)]
        cols[a.lower()] = s
    return pd.DataFrame(cols).sort_index()
