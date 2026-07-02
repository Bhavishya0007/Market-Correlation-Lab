"""Global Market Correlation Lab — Streamlit app.

Run with:  streamlit run app.py
Optional:  export ALPHAVANTAGE_API_KEY=your_key   (enables macro + news tabs)
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import data_sources as ds
import mcp_news

st.set_page_config(page_title="Global Market Correlation Lab",
                   page_icon="🌐", layout="wide")

# One quiet palette used everywhere: ink for developed, saffron for emerging.
COLORS = {"line_a": "#1f3a5f", "line_b": "#d4691e", "accent": "#6b7f94"}

# ---------------------------------------------------------------------------
# Sidebar — the whole analysis is parameterized here
# ---------------------------------------------------------------------------

st.sidebar.title("Analysis settings")

all_names = list(ds.INDICES.keys())
selected = st.sidebar.multiselect(
    "Markets", all_names,
    default=["S&P 500 (US)", "Nifty 50 (India)", "FTSE 100 (UK)",
             "DAX (Germany)", "Nikkei 225 (Japan)", "Hang Seng (HK)",
             "Bovespa (Brazil)"],
)
start = st.sidebar.date_input("Start date", date(2010, 1, 1))
end = st.sidebar.date_input("End date", date.today())
freq = st.sidebar.radio("Return frequency", ["Daily", "Weekly", "Monthly"],
                        index=1,
                        help="Weekly is recommended across time zones: India "
                             "closes before the US opens, so same-day daily "
                             "correlations understate the real linkage.")
usd_terms = st.sidebar.toggle("Convert to USD terms", value=True,
                              help="A USD allocator's view: index return plus "
                                   "the currency move. Turn off for local-"
                                   "currency correlations.")
pair_a = st.sidebar.selectbox("Pair: market A", selected, index=0)
pair_b = st.sidebar.selectbox("Pair: market B", selected,
                              index=min(1, len(selected) - 1))
window_label = st.sidebar.select_slider(
    "Rolling window", options=["6 months", "1 year", "2 years", "3 years"],
    value="1 year")

WINDOW_PERIODS = {
    "Daily":   {"6 months": 126, "1 year": 252, "2 years": 504, "3 years": 756},
    "Weekly":  {"6 months": 26,  "1 year": 52,  "2 years": 104, "3 years": 156},
    "Monthly": {"6 months": 6,   "1 year": 12,  "2 years": 24,  "3 years": 36},
}
window = WINDOW_PERIODS[freq][window_label]

if len(selected) < 2:
    st.info("Pick at least two markets in the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

with st.spinner("Loading price history..."):
    prices = ds.load_index_prices(tuple(selected), start, end)
    if usd_terms:
        fx = ds.load_fx(start, end)
        prices = ds.to_usd(prices, fx)
    risk = ds.load_risk_series(start, end)

returns = an.log_returns(prices, freq)

st.title("Global Market Correlation Lab")
st.caption(
    f"{freq} log returns, {'USD' if usd_terms else 'local-currency'} terms, "
    f"{prices.index.min():%b %Y} – {prices.index.max():%b %Y}."
)

tab_matrix, tab_pair, tab_macro, tab_news = st.tabs(
    ["Correlation matrix", f"Deep dive: {pair_a} vs {pair_b}",
     "Macro & risk lens", "Market news (MCP)"]
)

# ---------------------------------------------------------------------------
# Tab 1 — cross-market correlation matrix + allocator's table
# ---------------------------------------------------------------------------

with tab_matrix:
    corr = an.correlation_matrix(returns)
    fig = px.imshow(corr, text_auto=".2f", zmin=-0.2, zmax=1.0,
                    color_continuous_scale="RdBu_r", aspect="auto")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("What this buys an allocator")
    st.dataframe(an.diversification_snapshot(returns, benchmark=pair_a),
                 use_container_width=True, hide_index=True)
    st.caption(
        "Correlation vs your chosen base market, with return and volatility "
        "for context. Low correlation only helps if the market also carries "
        "its own return — a diversifier that never goes up is just drag."
    )

# ---------------------------------------------------------------------------
# Tab 2 — the pair deep dive (this is the US-vs-India story)
# ---------------------------------------------------------------------------

with tab_pair:
    roll = an.rolling_correlation(returns, pair_a, pair_b, window)
    beta = an.rolling_beta(returns, pair_b, pair_a, window)

    col1, col2, col3 = st.columns(3)
    full = returns[[pair_a, pair_b]].dropna()
    col1.metric("Full-sample correlation",
                f"{full[pair_a].corr(full[pair_b]):.2f}")
    if roll.dropna().size:
        col2.metric(f"Latest {window_label} correlation",
                    f"{roll.dropna().iloc[-1]:.2f}")
    if beta.dropna().size:
        col3.metric(f"Latest beta of {pair_b} on {pair_a}",
                    f"{beta.dropna().iloc[-1]:.2f}")

    fig = go.Figure()
    fig.add_scatter(x=roll.index, y=roll, name="Rolling correlation",
                    line=dict(color=COLORS["line_a"], width=2))
    fig.add_scatter(x=beta.index, y=beta, name=f"Rolling beta ({pair_b} on {pair_a})",
                    line=dict(color=COLORS["line_b"], width=1.5, dash="dot"))
    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Correlation says whether the two move together; beta says how much. "
        "A correlation of 0.6 with a beta of 1.3 is a very different risk "
        "position than 0.6 with a beta of 0.7."
    )

    st.subheader("Era comparison")
    default_breaks = "2016-11-08, 2020-03-01, 2022-03-16"
    breaks_text = st.text_input(
        "Era breakpoints (comma-separated dates)", default_breaks,
        help="Defaults: demonetization/US election (Nov 2016), COVID crash "
             "(Mar 2020), start of the Fed hiking cycle (Mar 2022).")
    try:
        breakpoints = [b.strip() for b in breaks_text.split(",") if b.strip()]
        eras = an.era_correlations(returns, pair_a, pair_b, breakpoints)
        st.dataframe(eras, use_container_width=True, hide_index=True)
    except Exception:
        st.warning("Couldn't parse those dates — use YYYY-MM-DD.")

# ---------------------------------------------------------------------------
# Tab 3 — macro & risk conditioning
# ---------------------------------------------------------------------------

with tab_macro:
    st.subheader("Is the correlation structural, or just risk-off?")
    regimes = an.vix_regime_correlations(returns, risk["VIX"], pair_a, pair_b)
    if not regimes.empty:
        fig = px.bar(regimes, x="Risk regime", y="Correlation",
                     text_auto=".2f", color="Risk regime",
                     color_discrete_sequence=[COLORS["line_a"],
                                              COLORS["accent"],
                                              COLORS["line_b"]])
        fig.update_layout(height=380, showlegend=False,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Everything correlates in a crisis. If the pair's correlation is "
            "elevated even in the calm tercile, that's evidence of a genuine "
            "structural shift (integrated rate cycle, foreign flows) rather "
            "than the usual stress-driven convergence."
        )

    st.subheader("Macro overlays")
    macro = ds.load_macro_bundle()
    if macro:
        chosen = st.multiselect("Series", list(macro.keys()),
                                default=list(macro.keys())[:2])
        if chosen:
            fig = go.Figure()
            roll = an.rolling_correlation(returns, pair_a, pair_b, window)
            fig.add_scatter(x=roll.index, y=roll,
                            name=f"{pair_a} vs {pair_b} rolling corr",
                            line=dict(color=COLORS["line_a"], width=2))
            for i, name in enumerate(chosen):
                s = macro[name].loc[str(start):]
                fig.add_scatter(x=s.index, y=s, name=name, yaxis="y2",
                                line=dict(width=1.5,
                                          dash="dot" if i else "dash"))
            fig.update_layout(
                height=440, margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(title="Correlation"),
                yaxis2=dict(title="Macro (%)", overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The 2022+ pattern to look for: a shared global rate/inflation "
                "shock is a common factor that mechanically raises cross-"
                "market correlation — one reason developed–emerging "
                "correlations rose when the Fed cycle started."
            )
    else:
        st.info(
            "Set ALPHAVANTAGE_API_KEY (env var or .streamlit/secrets.toml) to "
            "load Fed funds, US CPI, and treasury yield overlays. Free keys: "
            "alphavantage.co/support/#api-key"
        )

# ---------------------------------------------------------------------------
# Tab 4 — news via the Alpha Vantage MCP server
# ---------------------------------------------------------------------------

with tab_news:
    st.subheader("Market news & sentiment")
    c1, c2, c3 = st.columns([2, 2, 1])
    tickers = c1.text_input("Tickers (optional, e.g. SPY, INDA)", "")
    topic = c2.selectbox("Topic", mcp_news.TOPIC_CHOICES, index=0)
    limit = c3.number_input("Items", 5, 50, 20)

    if st.button("Refresh news (clear cache)"):
        mcp_news.fetch_news.clear()

    try:
        news, source = mcp_news.fetch_news(tickers.strip(), topic, int(limit))
    except mcp_news.NewsFetchError as err:
        st.error(str(err))
        st.caption("Tip: the free Alpha Vantage tier allows 25 requests/day, "
                   "shared with the macro tab. Rate-limit notices appear "
                   "here verbatim.")
        news, source = pd.DataFrame(), ""

    if not news.empty:
        st.caption(f"Source transport: "
                   f"{'MCP server' if source == 'mcp' else 'REST fallback'}")
        for _, row in news.iterrows():
            with st.container(border=True):
                st.markdown(f"**[{row['Title']}]({row['URL']})**")
                score = f" · score {row['Score']:+.2f}" if pd.notna(row["Score"]) else ""
                st.caption(f"{row['Source']} · {row['Published']:%d %b %Y %H:%M} "
                           f"· {row['Sentiment']}{score}")
                if row["Summary"]:
                    st.write(row["Summary"])
