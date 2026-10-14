# Data

`btc.csv` and `eth.csv` hold one row per calendar day: `date,close`, where
`close` is the Coin Metrics daily USD reference price (`PriceUSD`, end of day UTC).

- **Source:** Coin Metrics community data archive, <https://github.com/coinmetrics/data>
- **License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), data © Coin Metrics.
  It is redistributed here unchanged apart from cleaning, for non-commercial use with attribution.
- **Cleaning** (`cryptobt.data.clean_prices`): sort, drop duplicate days and non-positive
  prices, reindex to a daily calendar, forward-fill gaps of up to 3 days.

Refresh the snapshot with:

```bash
python -m cryptobt fetch --assets btc eth
```

The backtests start in 2014 (BTC) and 2016 (ETH) by default, because earlier prices
come from very thin markets that could not have been traded at those quotes.
