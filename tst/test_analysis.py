"""Tests for analysis.py — the statistical core.

Strategy: build synthetic price/return series where the true correlation
or beta is known by construction, then assert the functions recover it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import analysis as an

RNG = np.random.default_rng(7)


def _price_frame(n_days: int = 900) -> pd.DataFrame:
    """Two random-walk price series over business days."""
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rets = RNG.normal(0, 0.01, size=(n_days, 2))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=["US", "India"])


def _correlated_returns(n: int, rho: float, freq: str = "W-FRI") -> pd.DataFrame:
    """Return frame with a known population correlation rho."""
    idx = pd.date_range("2015-01-02", periods=n, freq=freq)
    x = RNG.normal(0, 0.02, n)
    eps = RNG.normal(0, 0.02, n)
    y = rho * x + np.sqrt(1 - rho ** 2) * eps
    return pd.DataFrame({"US": x, "India": y}, index=idx)


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------

def test_log_returns_daily_values():
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=idx)
    rets = an.log_returns(prices, "Daily")
    assert len(rets) == 2
    assert np.isclose(rets["A"].iloc[0], np.log(110 / 100))
    assert np.isclose(rets["A"].iloc[1], np.log(99 / 110))


def test_log_returns_weekly_downsamples():
    prices = _price_frame(500)
    daily = an.log_returns(prices, "Daily")
    weekly = an.log_returns(prices, "Weekly")
    monthly = an.log_returns(prices, "Monthly")
    assert len(weekly) < len(daily)
    assert len(monthly) < len(weekly)


def test_log_returns_are_additive_across_frequencies():
    # Log returns should sum: total daily return == total weekly return.
    prices = _price_frame(500)
    daily_total = an.log_returns(prices, "Daily")["US"].sum()
    weekly_total = an.log_returns(prices, "Weekly")["US"].sum()
    # Same start/end prices up to resampling edge effects at the boundaries.
    assert abs(daily_total - weekly_total) < 0.05


# ---------------------------------------------------------------------------
# correlation_matrix / rolling_correlation
# ---------------------------------------------------------------------------

def test_correlation_matrix_recovers_known_rho():
    rets = _correlated_returns(3000, rho=0.6)
    corr = an.correlation_matrix(rets)
    assert corr.loc["US", "US"] == 1.0
    assert abs(corr.loc["US", "India"] - 0.6) < 0.05


def test_correlation_matrix_perfect_and_inverse():
    idx = pd.date_range("2020-01-03", periods=100, freq="W-FRI")
    x = RNG.normal(0, 0.02, 100)
    rets = pd.DataFrame({"A": x, "B": 2 * x, "C": -x}, index=idx)
    corr = an.correlation_matrix(rets)
    assert np.isclose(corr.loc["A", "B"], 1.0)
    assert np.isclose(corr.loc["A", "C"], -1.0)


def test_rolling_correlation_detects_regime_change():
    # First half rho ~ 0, second half rho ~ 0.9: the rolling series should
    # end far above where it started.
    low = _correlated_returns(300, rho=0.0)
    high = _correlated_returns(300, rho=0.9)
    high.index = pd.date_range(low.index[-1] + pd.Timedelta(weeks=1),
                               periods=300, freq="W-FRI")
    rets = pd.concat([low, high])
    roll = an.rolling_correlation(rets, "US", "India", window=52).dropna()
    assert roll.iloc[:50].mean() < 0.35
    assert roll.iloc[-50:].mean() > 0.7


def test_rolling_correlation_bounds():
    rets = _correlated_returns(500, rho=0.5)
    roll = an.rolling_correlation(rets, "US", "India", window=26).dropna()
    assert (roll <= 1.0 + 1e-9).all() and (roll >= -1.0 - 1e-9).all()


# ---------------------------------------------------------------------------
# rolling_beta
# ---------------------------------------------------------------------------

def test_rolling_beta_recovers_known_beta():
    idx = pd.date_range("2015-01-02", periods=2000, freq="W-FRI")
    bench = pd.Series(RNG.normal(0, 0.02, 2000), index=idx)
    market = 1.4 * bench + RNG.normal(0, 0.005, 2000)
    rets = pd.DataFrame({"Bench": bench, "Mkt": market})
    beta = an.rolling_beta(rets, "Mkt", "Bench", window=104).dropna()
    assert abs(beta.mean() - 1.4) < 0.05


def test_beta_differs_from_correlation_when_vols_differ():
    # Perfect correlation, but market moves 2x the benchmark: corr = 1, beta = 2.
    idx = pd.date_range("2020-01-03", periods=200, freq="W-FRI")
    bench = pd.Series(RNG.normal(0, 0.02, 200), index=idx)
    rets = pd.DataFrame({"Bench": bench, "Mkt": 2 * bench})
    beta = an.rolling_beta(rets, "Mkt", "Bench", window=52).dropna()
    corr = an.rolling_correlation(rets, "Bench", "Mkt", 52).dropna()
    assert np.isclose(beta.iloc[-1], 2.0)
    assert np.isclose(corr.iloc[-1], 1.0)


# ---------------------------------------------------------------------------
# era_correlations
# ---------------------------------------------------------------------------

def test_era_correlations_splits_and_measures():
    low = _correlated_returns(300, rho=0.1)
    high = _correlated_returns(300, rho=0.8)
    high.index = pd.date_range(low.index[-1] + pd.Timedelta(weeks=1),
                               periods=300, freq="W-FRI")
    rets = pd.concat([low, high])
    breakpoint = str(high.index[0].date())
    eras = an.era_correlations(rets, "US", "India", [breakpoint])
    assert len(eras) == 2
    assert eras["Correlation"].iloc[0] < eras["Correlation"].iloc[1]
    assert (eras["Observations"] >= 20).all()


def test_era_correlations_skips_tiny_eras():
    rets = _correlated_returns(100, rho=0.5)
    # Breakpoint 5 rows before the end -> second era too small to report.
    bp = str(rets.index[-5].date())
    eras = an.era_correlations(rets, "US", "India", [bp])
    assert len(eras) == 1


# ---------------------------------------------------------------------------
# vix_regime_correlations
# ---------------------------------------------------------------------------

def test_vix_regimes_capture_stress_convergence():
    # Construct returns whose correlation is high exactly when VIX is high.
    n = 900
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    vix_level = np.where(np.arange(n) % 3 == 0, 35.0, 14.0)  # stressed 1/3 of weeks
    x = RNG.normal(0, 0.02, n)
    eps = RNG.normal(0, 0.02, n)
    rho = np.where(vix_level > 30, 0.9, 0.1)
    y = rho * x + np.sqrt(1 - rho ** 2) * eps
    rets = pd.DataFrame({"US": x, "India": y}, index=idx)
    vix = pd.Series(vix_level, index=idx)

    regimes = an.vix_regime_correlations(rets, vix, "US", "India", n_buckets=3)
    assert not regimes.empty
    calm = regimes.loc[regimes["Risk regime"].str.startswith("Calm"), "Correlation"]
    stressed = regimes.loc[regimes["Risk regime"].str.startswith("Stressed"), "Correlation"]
    if len(calm) and len(stressed):
        assert stressed.iloc[0] > calm.iloc[0]


def test_vix_regimes_columns():
    rets = _correlated_returns(300, rho=0.4)
    vix = pd.Series(RNG.uniform(12, 40, 300), index=rets.index)
    regimes = an.vix_regime_correlations(rets, vix, "US", "India")
    assert set(regimes.columns) == {"Risk regime", "Avg VIX", "Correlation",
                                    "Observations"}


# ---------------------------------------------------------------------------
# diversification_snapshot + helpers
# ---------------------------------------------------------------------------

def test_diversification_snapshot_excludes_benchmark_and_sorts():
    idx = pd.date_range("2018-01-05", periods=400, freq="W-FRI")
    x = RNG.normal(0, 0.02, 400)
    rets = pd.DataFrame({
        "US": x,
        "Clone": x + RNG.normal(0, 0.002, 400),          # corr ~ 1
        "Random": RNG.normal(0, 0.02, 400),               # corr ~ 0
    }, index=idx)
    snap = an.diversification_snapshot(rets, benchmark="US")
    assert "US" not in snap["Market"].values
    corrs = snap["Corr vs US"].tolist()
    assert corrs == sorted(corrs)
    labels = dict(zip(snap["Market"], snap["Diversification value"]))
    assert labels["Clone"] == "Little diversification"
    assert labels["Random"] == "Strong diversifier"


def test_diversification_labels_boundaries():
    assert an._diversification_label(0.1) == "Strong diversifier"
    assert an._diversification_label(0.4) == "Moderate diversifier"
    assert an._diversification_label(0.6) == "Weak diversifier"
    assert an._diversification_label(0.9) == "Little diversification"


def test_annualization_factor_by_frequency():
    daily = pd.bdate_range("2024-01-01", periods=50)
    weekly = pd.date_range("2024-01-05", periods=50, freq="W-FRI")
    monthly = pd.date_range("2024-01-31", periods=50, freq="ME")
    assert an._annualization_factor(daily) == 252.0
    assert an._annualization_factor(weekly) == 52.0
    assert an._annualization_factor(monthly) == 12.0
