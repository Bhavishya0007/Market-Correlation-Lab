"""Tests for data_sources.py — the pure parts (FX conversion, payload parsing).

Network-touching functions (yfinance downloads, HTTP) are intentionally not
exercised here; their pure cores are what carry the logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_sources as ds


def test_to_usd_converts_local_indices():
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({
        "S&P 500 (US)": [100.0, 101.0, 102.0],
        "Nifty 50 (India)": [20000.0, 20200.0, 20400.0],
    }, index=idx)
    fx = pd.DataFrame({
        "USD": [1.0, 1.0, 1.0],
        "INR": [1 / 83.0, 1 / 84.0, 1 / 85.0],   # USD per 1 INR
        "GBP": [1.27, 1.27, 1.27],
        "EUR": [1.08, 1.08, 1.08],
        "JPY": [1 / 150.0] * 3,
        "HKD": [1 / 7.8] * 3,
        "CNY": [1 / 7.2] * 3,
        "BRL": [1 / 5.0] * 3,
    }, index=idx)

    usd = ds.to_usd(prices, fx)
    # USD index unchanged; INR index divided by USDINR.
    assert np.allclose(usd["S&P 500 (US)"], prices["S&P 500 (US)"])
    assert np.isclose(usd["Nifty 50 (India)"].iloc[0], 20000.0 / 83.0)
    assert np.isclose(usd["Nifty 50 (India)"].iloc[2], 20400.0 / 85.0)


def test_to_usd_ffills_missing_fx_days():
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"Nifty 50 (India)": [100.0, 100.0, 100.0]}, index=idx)
    fx = pd.DataFrame({"INR": [0.012], "USD": [1.0], "GBP": [1.27],
                       "EUR": [1.08], "JPY": [0.0066], "HKD": [0.128],
                       "CNY": [0.139], "BRL": [0.2]},
                      index=[idx[0]])  # FX only on day 1
    usd = ds.to_usd(prices, fx)
    assert np.allclose(usd["Nifty 50 (India)"], 100.0 * 0.012)


def test_currency_depreciation_lowers_usd_returns():
    # Flat local index + depreciating INR must produce negative USD returns.
    idx = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"Nifty 50 (India)": [100.0] * 5}, index=idx)
    fx = pd.DataFrame({"INR": [0.0125, 0.0124, 0.0123, 0.0122, 0.0121],
                       "USD": [1.0] * 5, "GBP": [1.27] * 5, "EUR": [1.08] * 5,
                       "JPY": [0.0066] * 5, "HKD": [0.128] * 5,
                       "CNY": [0.139] * 5, "BRL": [0.2] * 5}, index=idx)
    usd = ds.to_usd(prices, fx)
    assert (usd["Nifty 50 (India)"].diff().dropna() < 0).all()


def test_parse_av_series_happy_path():
    payload = {
        "name": "Federal Funds Rate",
        "data": [
            {"date": "2024-03-01", "value": "5.33"},
            {"date": "2024-02-01", "value": "5.33"},
            {"date": "2024-01-01", "value": "5.33"},
        ],
    }
    s = ds.parse_av_series(payload)
    assert s is not None
    assert len(s) == 3
    assert s.index.is_monotonic_increasing
    assert np.isclose(s.iloc[0], 5.33)
    assert s.name == "Federal Funds Rate"


def test_parse_av_series_skips_placeholder_values():
    payload = {"data": [
        {"date": "2024-01-01", "value": "3.1"},
        {"date": "2024-02-01", "value": "."},
        {"date": "2024-03-01", "value": ""},
        {"date": "2024-04-01", "value": None},
    ]}
    s = ds.parse_av_series(payload)
    assert s is not None and len(s) == 1


def test_parse_av_series_handles_error_payloads():
    assert ds.parse_av_series({}) is None
    assert ds.parse_av_series({"Information": "rate limit reached"}) is None
    assert ds.parse_av_series({"data": []}) is None
    assert ds.parse_av_series({"data": [{"date": "2024-01-01", "value": "."}]}) is None


def test_index_universe_is_consistent():
    # Every configured index has a currency the FX table can supply.
    supplied = set(ds.FX_TICKERS) | {"USD"}
    for name, meta in ds.INDICES.items():
        assert meta["currency"] in supplied, name
        assert meta["bucket"] in {"Developed", "Emerging"}
