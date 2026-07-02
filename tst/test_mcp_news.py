"""Tests for mcp_news.py — error extraction and feed parsing.

The MCP/REST transports themselves are not exercised (network); the tests
cover the logic that decides success vs failure and shapes the output.
"""

from __future__ import annotations

import pandas as pd

import mcp_news


def test_server_error_extraction():
    assert mcp_news._server_error({"Error Message": "invalid key"}) == "invalid key"
    assert mcp_news._server_error({"Information": "rate limit"}) == "rate limit"
    assert mcp_news._server_error({"Note": "premium endpoint"}) == "premium endpoint"
    assert mcp_news._server_error({"feed": [{}]}) is None
    assert mcp_news._server_error({}) is None


def test_feed_to_df_shapes_and_sorts():
    feed = [
        {"time_published": "20260101T090000", "title": "Old story",
         "source": "Reuters", "overall_sentiment_label": "Neutral",
         "overall_sentiment_score": "0.05", "url": "https://a", "summary": "s1"},
        {"time_published": "20260630T120000", "title": "New story",
         "source": "Bloomberg", "overall_sentiment_label": "Bullish",
         "overall_sentiment_score": "0.41", "url": "https://b", "summary": "s2"},
    ]
    df = mcp_news.feed_to_df(feed)
    assert list(df.columns) == ["Published", "Title", "Source", "Sentiment",
                                "Score", "URL", "Summary"]
    assert df["Title"].iloc[0] == "New story"          # newest first
    assert abs(df["Score"].iloc[0] - 0.41) < 1e-9      # numeric, not string


def test_feed_to_df_tolerates_malformed_items():
    feed = [
        {"title": "No timestamp at all"},
        {"time_published": "not-a-date", "title": "Bad timestamp",
         "overall_sentiment_score": "not-a-number"},
    ]
    df = mcp_news.feed_to_df(feed)
    assert len(df) == 2
    assert df["Published"].isna().all()
    assert df["Score"].isna().all()


def test_fetch_news_raises_without_key(monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    try:
        mcp_news.fetch_news("", "financial_markets", 5)
    except mcp_news.NewsFetchError as err:
        assert "API key" in str(err)
    else:
        raise AssertionError("expected NewsFetchError when no key is set")


def test_fetch_news_surfaces_server_message(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")

    async def fake_mcp(key, tickers, topics, limit):
        return {"Information": "rate limit reached for today"}

    monkeypatch.setattr(mcp_news, "_fetch_via_mcp", fake_mcp)
    try:
        # Vary args so a cached success from another test can't be returned.
        mcp_news.fetch_news("TESTTICKER", "economy_macro", 7)
    except mcp_news.NewsFetchError as err:
        assert "rate limit" in str(err)
    else:
        raise AssertionError("expected NewsFetchError on server error payload")
