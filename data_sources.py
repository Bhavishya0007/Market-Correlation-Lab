"""Data layer for the Global Market Correlation app.

Price history comes from Yahoo Finance (via yfinance) because correlation
work needs 10-15 years of daily closes across many indices, which free
Alpha Vantage rate limits can't serve. Macro series (Fed funds, CPI,
treasury yields) come from Alpha Vantage's REST API. News comes from
Alpha Vantage's MCP server (see mcp_news.py).
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Universe: a developed / emerging split with the US-India pair front and center
# ---------------------------------------------------------------------------

INDICES: dict[str, dict] = {
    "S&P 500 (US)":        {"ticker": "^GSPC",     "currency": "USD", "bucket": "Developed"},
    "Nasdaq 100 (US)":     {"ticker": "^NDX",      "currency": "USD", "bucket": "Developed"},
    "Nifty 50 (India)":    {"ticker": "^NSEI",     "currency": "INR", "bucket": "Emerging"},
    "Sensex (India)":      {"ticker": "^BSESN",    "currency": "INR", "bucket": "Emerging"},
    "FTSE 100 (UK)":       {"ticker": "^FTSE",     "currency": "GBP", "bucket": "Developed"},
    "DAX (Germany)":       {"ticker": "^GDAXI",    "currency": "EUR", "bucket": "Developed"},
    "Nikkei 225 (Japan)":  {"ticker": "^N225",     "currency": "JPY", "bucket": "Developed"},
    "Hang Seng (HK)":      {"ticker": "^HSI",      "currency": "HKD", "bucket": "Emerging"},
    "Shanghai Comp (CN)":  {"ticker": "000001.SS", "currency": "CNY", "bucket": "Emerging"},
    "Bovespa (Brazil)":    {"ticker": "^BVSP",     "currency": "BRL", "bucket": "Emerging"},
}

# USD per 1 unit of local currency is derived from these Yahoo FX tickers
FX_TICKERS = {
    "INR": "INR=X",   # USD/INR quoted as INR per USD -> invert
    "GBP": "GBPUSD=X",
    "EUR": "EURUSD=X",
    "JPY": "JPY=X",
    "HKD": "HKD=X",
    "CNY": "CNY=X",
    "BRL": "BRL=X",
}
FX_QUOTED_PER_USD = {"INR=X", "JPY=X", "HKD=X", "CNY=X", "BRL=X"}

RISK_TICKERS = {
    "VIX": "^VIX",          # equity risk aversion proxy
    "US 10Y yield": "^TNX",  # quoted as yield * 10
}

ALPHAVANTAGE_BASE = "https://www.alphavantage.co/query"


def alphavantage_key() -> str | None:
    """API key from env var or Streamlit secrets, if configured."""
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ALPHAVANTAGE_API_KEY"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_index_prices(names: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Daily closes for the selected indices, one column per index name."""
    tickers = [INDICES[n]["ticker"] for n in names]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    ticker_to_name = {INDICES[n]["ticker"]: n for n in names}
    closes = closes.rename(columns=ticker_to_name)
    return closes[list(names)].sort_index()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_fx(start: date, end: date) -> pd.DataFrame:
    """USD value of 1 unit of each local currency, one column per currency code."""
    raw = yf.download(list(FX_TICKERS.values()), start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    out = {}
    for ccy, tkr in FX_TICKERS.items():
        series = raw[tkr]
        out[ccy] = 1.0 / series if tkr in FX_QUOTED_PER_USD else series
    fx = pd.DataFrame(out).sort_index()
    fx["USD"] = 1.0
    return fx


def to_usd(prices: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Convert each index column into USD terms using its local currency."""
    fx_aligned = fx.reindex(prices.index).ffill()
    converted = {}
    for name in prices.columns:
        ccy = INDICES[name]["currency"]
        converted[name] = prices[name] * fx_aligned[ccy]
    return pd.DataFrame(converted)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_risk_series(start: date, end: date) -> pd.DataFrame:
    """VIX level and US 10Y yield (in %) as daily series."""
    raw = yf.download(list(RISK_TICKERS.values()), start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    df = pd.DataFrame({
        "VIX": raw["^VIX"],
        "US 10Y yield": raw["^TNX"] / 10.0,
    }).sort_index()
    return df


# ---------------------------------------------------------------------------
# Macro series from Alpha Vantage (needs a free API key)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_av_macro(function: str, extra: dict | None = None) -> pd.Series | None:
    """Fetch one Alpha Vantage economic series (e.g. CPI, FEDERAL_FUNDS_RATE,
    TREASURY_YIELD). Returns a date-indexed Series, or None if no key /
    request failed."""
    key = alphavantage_key()
    if not key:
        return None
    params = {"function": function, "apikey": key, "datatype": "json"}
    if extra:
        params.update(extra)
    try:
        resp = requests.get(ALPHAVANTAGE_BASE, params=params, timeout=30)
        payload = resp.json()
        rows = payload.get("data", [])
        if not rows:
            return None
        s = pd.Series(
            {pd.Timestamp(r["date"]): float(r["value"])
             for r in rows if r.get("value") not in (None, ".", "")},
            name=payload.get("name", function),
        ).sort_index()
        return s
    except Exception:
        return None


def load_macro_bundle() -> dict[str, pd.Series]:
    """The macro overlays used by the 'Macro & Risk Lens' tab."""
    bundle = {}
    fed = load_av_macro("FEDERAL_FUNDS_RATE", {"interval": "monthly"})
    if fed is not None:
        bundle["Fed funds rate (%)"] = fed
    cpi = load_av_macro("CPI", {"interval": "monthly"})
    if cpi is not None:
        bundle["US CPI YoY (%)"] = cpi.pct_change(12) * 100
    y10 = load_av_macro("TREASURY_YIELD", {"interval": "monthly", "maturity": "10year"})
    if y10 is not None:
        bundle["US 10Y yield, monthly (%)"] = y10
    return bundle
