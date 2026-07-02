"""Correlation and regime analytics.

Everything operates on log returns. Weekly returns are the default for
cross-market work because Asia and the US don't share a trading day:
same-calendar-day daily correlations understate the true linkage (India
closes before the US opens), and weekly aggregation largely fixes the
asynchronicity without needing lead-lag adjustments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FREQ_RULES = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}


def log_returns(prices: pd.DataFrame, freq: str = "Weekly") -> pd.DataFrame:
    """Log returns at the requested frequency, dropping all-NaN rows."""
    rule = FREQ_RULES[freq]
    px = prices if rule is None else prices.resample(rule).last()
    rets = np.log(px / px.shift(1))
    return rets.dropna(how="all")


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation using pairwise-complete observations."""
    return returns.corr(min_periods=20)


def rolling_correlation(returns: pd.DataFrame, a: str, b: str, window: int) -> pd.Series:
    """Rolling correlation between two markets over `window` periods."""
    pair = returns[[a, b]].dropna()
    return pair[a].rolling(window, min_periods=max(10, window // 2)).corr(pair[b])


def rolling_beta(returns: pd.DataFrame, market: str, benchmark: str, window: int) -> pd.Series:
    """Rolling beta of `market` on `benchmark` — correlation with the
    volatility ratio baked in, so it answers 'how much does India move
    per 1% move in the US', not just 'do they move together'."""
    pair = returns[[market, benchmark]].dropna()
    cov = pair[market].rolling(window).cov(pair[benchmark])
    var = pair[benchmark].rolling(window).var()
    return cov / var


def era_correlations(returns: pd.DataFrame, a: str, b: str,
                     breakpoints: list[str]) -> pd.DataFrame:
    """Full-sample correlation of a pair inside each era defined by the
    breakpoint dates. Used to show the pre/post-2020 regime shift."""
    pair = returns[[a, b]].dropna()
    edges = [pair.index.min()] + [pd.Timestamp(bp) for bp in breakpoints] + [pair.index.max()]
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        chunk = pair.loc[start:end]
        if len(chunk) >= 20:
            rows.append({
                "Era": f"{start:%b %Y} – {end:%b %Y}",
                "Correlation": chunk[a].corr(chunk[b]),
                "Observations": len(chunk),
            })
    return pd.DataFrame(rows)


def vix_regime_correlations(returns: pd.DataFrame, vix: pd.Series,
                            a: str, b: str, n_buckets: int = 3) -> pd.DataFrame:
    """Correlation of the pair conditioned on the risk regime.

    VIX is averaged over each return period, then bucketed into terciles
    (calm / normal / stressed). If correlation only rises in the stressed
    bucket, the 'India has decoupled' story is really just the usual
    everything-falls-together crisis effect. If it has risen in the calm
    bucket too, the shift is structural.
    """
    # Align VIX to the return index: trailing 5-day average at each return date.
    vix = vix.dropna()
    period_vix = vix.reindex(
        pd.date_range(vix.index.min(), vix.index.max(), freq="D")
    ).ffill()
    pair = returns[[a, b]].dropna()
    avg_vix = pd.Series(
        [period_vix.loc[:d].tail(5).mean() for d in pair.index], index=pair.index
    )
    labels = ["Calm (low VIX)", "Normal", "Stressed (high VIX)"][:n_buckets]
    buckets = pd.qcut(avg_vix, n_buckets, labels=labels)
    rows = []
    for label in labels:
        chunk = pair[buckets == label]
        if len(chunk) >= 20:
            rows.append({
                "Risk regime": label,
                "Avg VIX": avg_vix[buckets == label].mean(),
                "Correlation": chunk[a].corr(chunk[b]),
                "Observations": len(chunk),
            })
    return pd.DataFrame(rows)


def diversification_snapshot(returns: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Per-market stats an allocator cares about, vs the chosen benchmark."""
    rows = []
    periods_per_year = _annualization_factor(returns.index)
    for name in returns.columns:
        if name == benchmark:
            continue
        pair = returns[[name, benchmark]].dropna()
        if len(pair) < 20:
            continue
        corr = pair[name].corr(pair[benchmark])
        vol = pair[name].std() * np.sqrt(periods_per_year)
        ann_ret = pair[name].mean() * periods_per_year
        rows.append({
            "Market": name,
            f"Corr vs {benchmark}": round(corr, 2),
            "Ann. return (log)": f"{ann_ret:.1%}",
            "Ann. volatility": f"{vol:.1%}",
            "Diversification value": _diversification_label(corr),
        })
    return pd.DataFrame(rows).sort_values(f"Corr vs {benchmark}")


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    median_gap = index.to_series().diff().median().days
    if median_gap <= 2:
        return 252.0
    if median_gap <= 8:
        return 52.0
    return 12.0


def _diversification_label(corr: float) -> str:
    if corr < 0.3:
        return "Strong diversifier"
    if corr < 0.55:
        return "Moderate diversifier"
    if corr < 0.75:
        return "Weak diversifier"
    return "Little diversification"
