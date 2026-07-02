"""Market news via the Alpha Vantage MCP server.

Alpha Vantage exposes its API as a remote MCP server at
    https://mcp.alphavantage.co/mcp?apikey=YOUR_KEY
This module connects as an MCP client (streamable HTTP transport) and calls
the NEWS_SENTIMENT tool. If the MCP dependency or server is unavailable it
falls back to the plain REST endpoint.

Failures raise NewsFetchError with the server's actual message (rate limit,
invalid key, etc.) so the UI can show it — and so Streamlit's cache never
stores a failed result.
"""

from __future__ import annotations

import asyncio
import json
import os

import pandas as pd
import requests
import streamlit as st

MCP_SERVER_URL = "https://mcp.alphavantage.co/mcp?apikey={key}"
REST_URL = "https://www.alphavantage.co/query"

TOPIC_CHOICES = [
    "financial_markets", "economy_monetary", "economy_macro",
    "economy_fiscal", "finance", "technology", "energy_transportation",
]

# Keys Alpha Vantage uses to report problems instead of returning a feed.
ERROR_KEYS = ("Error Message", "Information", "Note")


class NewsFetchError(Exception):
    """Raised when no usable feed came back; message explains why."""


def _get_key() -> str | None:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ALPHAVANTAGE_API_KEY"]
    except Exception:
        return None


def _server_error(payload: dict) -> str | None:
    """Extract Alpha Vantage's error/notice text from a payload, if any."""
    for k in ERROR_KEYS:
        if payload.get(k):
            return str(payload[k])
    return None


async def _fetch_via_mcp(key: str, tickers: str, topics: str, limit: int) -> dict:
    """Open an MCP session and call the NEWS_SENTIMENT tool."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    url = MCP_SERVER_URL.format(key=key)
    args: dict = {"limit": str(limit), "sort": "LATEST"}
    if tickers:
        args["tickers"] = tickers
    if topics:
        args["topics"] = topics

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("NEWS_SENTIMENT", arguments=args)
            for block in result.content:
                if getattr(block, "type", None) == "text":
                    return json.loads(block.text)
    return {}


def _fetch_via_rest(key: str, tickers: str, topics: str, limit: int) -> dict:
    params = {"function": "NEWS_SENTIMENT", "apikey": key,
              "limit": limit, "sort": "LATEST"}
    if tickers:
        params["tickers"] = tickers
    if topics:
        params["topics"] = topics
    resp = requests.get(REST_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_news(tickers: str = "", topics: str = "financial_markets",
               limit: int = 25) -> tuple[pd.DataFrame, str]:
    """Return (news dataframe, transport label).

    Raises NewsFetchError on any failure. Streamlit's cache does not store
    results when the function raises, so failures are always retried on the
    next run instead of being cached for 15 minutes.
    """
    key = _get_key()
    if not key:
        raise NewsFetchError(
            "No API key found. Set ALPHAVANTAGE_API_KEY as an environment "
            "variable in the shell running streamlit, or add it to "
            ".streamlit/secrets.toml.")

    payload, source, mcp_err = {}, "mcp", None
    try:
        payload = asyncio.run(_fetch_via_mcp(key, tickers, topics, limit))
    except Exception as exc:
        mcp_err = f"{type(exc).__name__}: {exc}"
        source = "rest"
        try:
            payload = _fetch_via_rest(key, tickers, topics, limit)
        except Exception as rest_exc:
            raise NewsFetchError(
                f"Both transports failed. MCP: {mcp_err} | "
                f"REST: {type(rest_exc).__name__}: {rest_exc}")

    server_msg = _server_error(payload)
    if server_msg:
        raise NewsFetchError(f"Alpha Vantage ({source}): {server_msg}")

    feed = payload.get("feed", [])
    if not feed:
        raise NewsFetchError(
            f"Empty feed from Alpha Vantage ({source}). Raw keys returned: "
            f"{list(payload.keys()) or 'none'}. Try removing ticker filters "
            f"or a broader topic.")

    rows = []
    for item in feed:
        rows.append({
            "Published": pd.to_datetime(item.get("time_published"),
                                        format="%Y%m%dT%H%M%S", errors="coerce"),
            "Title": item.get("title", ""),
            "Source": item.get("source", ""),
            "Sentiment": item.get("overall_sentiment_label", ""),
            "Score": pd.to_numeric(item.get("overall_sentiment_score"),
                                   errors="coerce"),
            "URL": item.get("url", ""),
            "Summary": item.get("summary", ""),
        })
    df = pd.DataFrame(rows).sort_values("Published", ascending=False)
    return df, source
