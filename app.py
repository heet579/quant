"""Interactive dashboard for quantitative equity research."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as A

st.set_page_config(
    page_title="Quant Finance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_pipeline():
    return A.run_full_pipeline()


def pct(x: float) -> str:
    return f"{x:.2%}"


def metric_row(metrics: dict, cols: list):
    c = st.columns(len(cols))
    for col, key in zip(c, cols):
        val = metrics[key]
        if key in ("Sharpe", "Sortino", "Calmar", "N Months"):
            col.metric(key, f"{val:.2f}" if key != "N Months" else str(int(val)))
        elif key == "Win Rate":
            col.metric(key, pct(val))
        else:
            col.metric(key, pct(val))


def cum_return_chart(series_dict: dict, title: str) -> go.Figure:
    fig = go.Figure()
    for label, ser in series_dict.items():
        s = ser.dropna()
        cum = (1 + s).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=label))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative return (base = 1)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Quant Research")
    st.caption("Equity alpha · ML signals · Risk analytics")
    st.markdown(
        f"""
        **Universe:** {len(A.TICKERS)} U.S. large caps  
        **Period:** {A.START[:4]}–{A.END[:4]}  
        **Benchmark:** SPY  

        **ML split**  
        Train → {A.TRAIN_END[:4]}  
        Val → {A.VAL_END[:4]}  
        Test → OOS 2024
        """
    )
    st.divider()
    run = st.button("↻ Re-run analysis", use_container_width=True)
    st.caption("First load downloads data & trains XGBoost (~1–2 min). Results are cached 6h.")

if run:
    load_pipeline.clear()

with st.spinner("Running pipeline — downloading prices, factors, ML model, backtest…"):
    R = load_pipeline()

st.success("Analysis ready.")

# ── Header ──────────────────────────────────────────────────────────────────
st.title("Quantitative Finance Analysis Dashboard")
st.markdown(
    "End-to-end **equity research**: exploratory analysis, cross-sectional factors, "
    "**XGBoost** return signals, transaction-cost-aware long/short backtest, and **VaR / stress** risk. "
    "All signals use strict **no-lookahead** rules."
)

tab_over, tab_eda, tab_fac, tab_ml, tab_port, tab_risk = st.tabs(
    ["Overview", "Market EDA", "Factors", "ML Model", "Portfolio", "Risk"]
)

# ── Overview ────────────────────────────────────────────────────────────────
with tab_over:
    m = R["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OOS Test IC", f"{m['test_ic']:.3f}")
    c2.metric("OOS Test R²", f"{m['test_r2']:.3f}")
    net_perf = R["perf"].loc["ML L/S Net"]
    c3.metric("L/S Net Sharpe", f"{net_perf['Sharpe']:.2f}")
    c4.metric("L/S Max Drawdown", pct(net_perf["Max Drawdown"]))

    st.plotly_chart(
        cum_return_chart(
            {
                "ML L/S Net": R["pnl"]["net"],
                "SPY": R["spy_sub"],
                "EW Universe": R["ew_sub"],
            },
            "Cumulative returns — ML long/short vs benchmarks (val + test)",
        ),
        use_container_width=True,
    )

    st.subheader("Performance summary (2023–2024 backtest)")
    perf_fmt = R["perf"].copy()
    for col in ["Ann. Return", "Ann. Vol", "Max Drawdown", "Win Rate"]:
        perf_fmt[col] = perf_fmt[col].map(lambda x: pct(x) if col != "Win Rate" else pct(x))
    for col in ["Sharpe", "Sortino", "Calmar"]:
        perf_fmt[col] = perf_fmt[col].map(lambda x: f"{x:.2f}")
    perf_fmt["N Months"] = perf_fmt["N Months"].astype(int)
    st.dataframe(perf_fmt, use_container_width=True, hide_index=False)

    st.subheader("Factor IC summary")
    st.dataframe(R["ic_table"], use_container_width=True, hide_index=True)

# ── EDA ─────────────────────────────────────────────────────────────────────
with tab_eda:
    rd = R["rets_daily"]
    col_a, col_b = st.columns(2)

    with col_a:
        picks = st.multiselect(
            "Tickers vs SPY",
            A.TICKERS,
            default=["NVDA", "AAPL", "MSFT", "JPM", "XOM"],
        )
        spy_cum = (1 + R["spy_prices"].pct_change().dropna()).cumprod()
        traces = {"SPY": spy_cum}
        for t in picks:
            traces[t] = (1 + rd[t]).cumprod()
        st.plotly_chart(cum_return_chart(traces, "Cumulative returns"), use_container_width=True)

    with col_b:
        port = rd.mean(axis=1)
        fig_hist = px.histogram(port, nbins=80, title="EW portfolio daily returns")
        fig_hist.update_layout(height=420, xaxis_title="Daily return", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Sector equal-weight annual returns")
    fig_heat = px.imshow(
        R["sector_annual"].T,
        labels=dict(x="Year", y="Sector", color="Return"),
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=-0.5,
        zmax=0.5,
    )
    fig_heat.update_layout(height=360, title="Annual sector returns")
    st.plotly_chart(fig_heat, use_container_width=True)

    corr = rd[A.TICKERS].corr()
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-0.2, zmax=1, title="Stock correlation matrix")
    fig_corr.update_layout(height=520)
    st.plotly_chart(fig_corr, use_container_width=True)

# ── Factors ─────────────────────────────────────────────────────────────────
with tab_fac:
    st.dataframe(R["ic_table"], use_container_width=True, hide_index=True)

    ic_choice = st.selectbox("Factor IC time series", list(R["ic_series"].keys()))
    ic = R["ic_series"][ic_choice]
    fig_ic = go.Figure()
    fig_ic.add_trace(go.Bar(x=ic.index, y=ic.values, name="Monthly IC", opacity=0.55))
    fig_ic.add_trace(go.Scatter(x=ic.index, y=ic.cumsum(), name="Cumulative IC", yaxis="y2", line=dict(width=2)))
    fig_ic.update_layout(
        title=f"{ic_choice} — monthly IC & cumulative IC",
        yaxis=dict(title="IC"),
        yaxis2=dict(title="Cum. IC", overlaying="y", side="right"),
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig_ic, use_container_width=True)

# ── ML ──────────────────────────────────────────────────────────────────────
with tab_ml:
    m = R["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Val IC", f"{m['val_ic']:.3f}")
    c2.metric("Test IC", f"{m['test_ic']:.3f}")
    c3.metric("Val R²", f"{m['val_r2']:.3f}")
    c4.metric("Best trees", str(m["best_iteration"]))

    imp = m["importance"].reset_index()
    imp.columns = ["Feature", "Importance"]
    fig_fi = px.bar(imp, x="Importance", y="Feature", orientation="h", title="XGBoost feature importance (gain)")
    fig_fi.update_layout(height=420, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.caption(
        "12 engineered features · trained 2019–2022 · validated 2023 · tested 2024. "
        "Monthly equity IC of 0.03–0.08 is typical in academic literature."
    )

# ── Portfolio ───────────────────────────────────────────────────────────────
with tab_port:
    st.plotly_chart(
        cum_return_chart(
            {
                "ML L/S Net": R["pnl"]["net"],
                "ML L/S Gross": R["pnl"]["gross"],
                "SPY": R["spy_sub"],
                "EW Universe": R["ew_sub"],
            },
            "Backtest cumulative returns",
        ),
        use_container_width=True,
    )

    st.subheader("ML L/S Net — monthly returns")
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in R["pnl"]["net"]]
    fig_bar = go.Figure(go.Bar(x=R["pnl"].index, y=R["pnl"]["net"], marker_color=colors))
    fig_bar.update_layout(height=320, yaxis_tickformat=".1%", title=f"Monthly net returns ({A.TC_BPS} bps one-way TC)")
    st.plotly_chart(fig_bar, use_container_width=True)

    pf = R["perf"].copy()
    for col in ["Ann. Return", "Ann. Vol", "Max Drawdown", "Win Rate"]:
        pf[col] = pf[col].map(pct)
    for col in ["Sharpe", "Sortino", "Calmar"]:
        pf[col] = pf[col].map(lambda x: f"{x:.2f}")
    st.dataframe(pf, use_container_width=True)

# ── Risk ────────────────────────────────────────────────────────────────────
with tab_risk:
    rows = []
    for alpha, conf, var_d in [(0.05, "95%", R["var_95"]), (0.01, "99%", R["var_99"])]:
        for method, v in var_d.items():
            rows.append({
                "Confidence": conf,
                "Method": method,
                "Daily VaR": pct(v["VaR"]),
                "Daily CVaR": pct(v["CVaR"]),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Drawdown — backtest period")
    def _dd(s):
        c = (1 + s.dropna()).cumprod()
        return c / c.cummax() - 1

    fig_dd = go.Figure()
    for label, ser in [("ML L/S Net", R["pnl"]["net"]), ("SPY", R["spy_sub"]), ("EW", R["ew_sub"])]:
        dd = _dd(ser)
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=label, fill="tozeroy"))
    fig_dd.update_layout(height=380, yaxis_tickformat=".0%", title="Underwater drawdown chart")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Historical stress scenarios (EW portfolio)")
    stress_rows = []
    for name, (s, e) in A.STRESS_SCENARIOS.items():
        mask = (R["port_daily"].index >= s) & (R["port_daily"].index <= e)
        r = R["port_daily"][mask]
        if len(r) == 0:
            continue
        cum = (1 + r).prod() - 1
        c = (1 + r).cumprod()
        dd = (c / c.cummax() - 1).min()
        stress_rows.append({"Scenario": name, "Days": len(r), "Return": pct(cum), "Max DD": pct(dd)})
    st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Research demo only — not investment advice. Data via yfinance. "
    "[GitHub](https://github.com/heet579/quant) · Built with Python, XGBoost, Streamlit, Plotly."
)
