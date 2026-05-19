"""Core quant pipeline — shared by notebook and Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

# ── Universe ──────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "JPM", "BAC", "GS", "MS", "WFC", "BLK",
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    "WMT", "HD", "COST", "MCD", "PG", "KO", "PEP",
    "XOM", "CVX", "COP",
    "CAT", "GE", "HON",
    "NEE", "DUK",
]

SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "AMZN": "Technology",
    "META": "Technology", "NVDA": "Technology", "TSLA": "Technology", "AMD": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "WFC": "Financials", "BLK": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare",
    "WMT": "Consumer", "HD": "Consumer", "COST": "Consumer", "MCD": "Consumer",
    "PG": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities",
}

SECTOR_COLORS = {
    "Technology": "#4C72B0", "Financials": "#DD8452", "Healthcare": "#55A868",
    "Consumer": "#C44E52", "Energy": "#8172B3", "Industrials": "#937860", "Utilities": "#DA8BC3",
}

START, END = "2019-01-01", "2024-12-31"
TRAIN_END, VAL_END = "2022-12-31", "2023-12-31"
TC_BPS = 10
RF_ANNUAL = 0.045

FEATURE_COLS = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "ret_24m",
    "mom_12_1", "vol_1m", "vol_3m", "vol_ratio",
    "price_52w_high", "cs_rank_mom", "cs_rank_vol",
]

STRESS_SCENARIOS = {
    "COVID Crash (Feb–Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 Rate Hike Selloff": ("2022-01-03", "2022-06-16"),
    "SVB / Banking Crisis (Mar 2023)": ("2023-03-06", "2023-03-24"),
    "Aug 2024 Vol Spike": ("2024-07-31", "2024-08-05"),
}


def _monthly_annual_freq() -> tuple[str, str]:
    test = pd.Series([1], index=pd.date_range("2020-01-01", periods=1, freq="D"))
    try:
        test.resample("ME").last()
        return "ME", "YE"
    except Exception:
        return "M", "Y"


MONTHLY_FREQ, ANNUAL_FREQ = _monthly_annual_freq()


def load_market_data() -> dict:
    """Download prices and return daily/monthly panels."""
    raw = yf.download(TICKERS + ["SPY"], start=START, end=END, auto_adjust=True, progress=False)
    prices_all = raw["Close"].ffill()

    spy_prices = prices_all["SPY"].dropna()
    prices = prices_all[TICKERS].dropna(how="all").ffill().dropna()
    rets_daily = prices.pct_change().dropna()

    prices_monthly = prices.resample(MONTHLY_FREQ).last()
    rets_monthly = prices_monthly.pct_change().dropna()
    spy_monthly = spy_prices.resample(MONTHLY_FREQ).last().pct_change().dropna()

    return {
        "prices": prices,
        "spy_prices": spy_prices,
        "rets_daily": rets_daily,
        "prices_monthly": prices_monthly,
        "rets_monthly": rets_monthly,
        "spy_monthly": spy_monthly,
    }


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True) - 0.5


def momentum_factor(pm: pd.DataFrame, lookback: int, skip: int = 1) -> pd.DataFrame:
    total_ret = pm.shift(skip) / pm.shift(skip + lookback) - 1
    return cs_rank(total_ret)


def low_vol_factor(rd: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    rolling_vol = rd.rolling(window).std().resample(MONTHLY_FREQ).last()
    return cs_rank(-rolling_vol)


def build_factors(pm: pd.DataFrame, rd: pd.DataFrame) -> dict:
    factor_mom121 = momentum_factor(pm, lookback=12, skip=1)
    factor_mom6 = momentum_factor(pm, lookback=6, skip=1)
    factor_reversal = -momentum_factor(pm, lookback=1, skip=0)
    factor_lowvol = low_vol_factor(rd)

    idx = (
        factor_mom121.index.intersection(factor_mom6.index)
        .intersection(factor_reversal.index)
        .intersection(factor_lowvol.index)
    )
    fwd_ret_1m = pm.pct_change().shift(-1)

    return {
        "mom121": factor_mom121.loc[idx],
        "mom6": factor_mom6.loc[idx],
        "reversal": factor_reversal.loc[idx],
        "lowvol": factor_lowvol.loc[idx],
        "fwd_ret_1m": fwd_ret_1m,
    }


def compute_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    common = factor.index.intersection(fwd_ret.index)
    ics = {}
    for dt in common:
        f = factor.loc[dt].dropna()
        r = fwd_ret.loc[dt].dropna()
        both = f.index.intersection(r.index)
        if len(both) < 10:
            continue
        ic, _ = spearmanr(f[both].values, r[both].values)
        ics[dt] = ic
    return pd.Series(ics)


def ic_summary(ic: pd.Series, name: str) -> dict:
    icir = ic.mean() / ic.std() if ic.std() > 0 else 0
    t_stat = icir * len(ic) ** 0.5
    return {
        "Factor": name,
        "Mean IC": round(ic.mean(), 4),
        "IC Std": round(ic.std(), 4),
        "ICIR": round(icir, 3),
        "t-stat": round(t_stat, 2),
        "% Positive": f"{(ic > 0).mean():.1%}",
    }


def build_ml_panel(pm: pd.DataFrame, rd: pd.DataFrame) -> pd.DataFrame:
    monthly_vol = rd.rolling(21).std().resample(MONTHLY_FREQ).last()
    records = []

    for tkr in pm.columns:
        p = pm[tkr].dropna()
        mv = monthly_vol[tkr].dropna()
        r = p.pct_change()

        df = pd.DataFrame(index=r.index)
        df["ret_1m"] = r
        df["ret_3m"] = p.pct_change(3)
        df["ret_6m"] = p.pct_change(6)
        df["ret_12m"] = p.pct_change(12)
        df["ret_24m"] = p.pct_change(24)
        df["mom_12_1"] = p.pct_change(12) - r
        df["vol_1m"] = mv
        df["vol_3m"] = mv.rolling(3).mean()
        df["vol_ratio"] = mv / mv.rolling(6).mean().clip(lower=1e-8)
        df["price_52w_high"] = p / p.rolling(12).max() - 1
        df["target"] = r.shift(-1)
        df["ticker"] = tkr
        records.append(df)

    panel = pd.concat(records).reset_index(names="date").dropna(subset=["ret_1m", "vol_1m"])

    for base_col, rank_col in [("mom_12_1", "cs_rank_mom"), ("vol_1m", "cs_rank_vol")]:
        pivot = panel.pivot(index="date", columns="ticker", values=base_col)
        ranked = (pivot.rank(axis=1, pct=True) - 0.5).stack(future_stack=True)
        ranked = ranked.reset_index(name=rank_col)
        ranked.columns = ["date", "ticker", rank_col]
        panel = panel.merge(ranked, on=["date", "ticker"], how="left")

    return panel.dropna(subset=FEATURE_COLS + ["target"])


def split_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = panel[panel["date"] <= TRAIN_END]
    val = panel[(panel["date"] > TRAIN_END) & (panel["date"] <= VAL_END)]
    test = panel[panel["date"] > VAL_END]
    return train, val, test


def train_xgb_model(X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def model_metrics(model, X_val, y_val, X_test, y_test) -> dict:
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    return {
        "val_ic": float(pd.Series(y_val.values).corr(pd.Series(y_val_pred))),
        "test_ic": float(pd.Series(y_test.values).corr(pd.Series(y_test_pred))),
        "val_r2": r2_score(y_val, y_val_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "val_rmse": mean_squared_error(y_val, y_val_pred) ** 0.5,
        "test_rmse": mean_squared_error(y_test, y_test_pred) ** 0.5,
        "best_iteration": getattr(model, "best_iteration", None),
        "importance": pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False),
    }


def generate_ls_weights(panel_sub: pd.DataFrame, model, top_q: float = 0.2) -> dict:
    panel_sub = panel_sub.copy()
    panel_sub["pred"] = model.predict(panel_sub[FEATURE_COLS])
    weights = {}
    for date, grp in panel_sub.groupby("date"):
        grp = grp.sort_values("pred")
        n = len(grp)
        k = max(1, int(n * top_q))
        longs = grp["ticker"].iloc[-k:].tolist()
        shorts = grp["ticker"].iloc[:k].tolist()
        w = {}
        for t in longs:
            w[t] = w.get(t, 0) + 1.0 / k
        for t in shorts:
            w[t] = w.get(t, 0) - 1.0 / k
        weights[date] = w
    return weights


def run_backtest(signals: dict, rets_monthly: pd.DataFrame, tc_bps: int = TC_BPS) -> pd.DataFrame:
    tc_rate = tc_bps / 10_000
    dates = sorted(signals.keys())
    ret_idx = rets_monthly.index
    results = []
    prev_w = {}

    for dt in dates:
        w = signals[dt]
        locs = ret_idx.searchsorted(dt)
        if locs >= len(ret_idx):
            continue
        if ret_idx[locs] <= dt:
            locs += 1
        if locs >= len(ret_idx):
            continue
        next_dt = ret_idx[locs]
        next_r = rets_monthly.loc[next_dt]

        gross = sum(w.get(t, 0) * next_r.get(t, 0) for t in set(w) | set(next_r.index))
        all_t = set(w) | set(prev_w)
        turnover = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_t)
        tc_cost = turnover * tc_rate

        results.append({"date": next_dt, "gross": gross, "tc": tc_cost, "net": gross - tc_cost})
        prev_w = w.copy()

    return pd.DataFrame(results).set_index("date")


def performance_metrics(rets: pd.Series, freq: int = 12, rf: float = RF_ANNUAL) -> dict:
    r = rets.dropna()
    ann = (1 + r).prod() ** (freq / len(r)) - 1
    vol = r.std() * freq ** 0.5
    sharpe = (ann - rf) / vol if vol > 0 else 0
    neg = r[r < 0]
    sortino = (ann - rf) / (neg.std() * freq ** 0.5) if len(neg) > 1 else 0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    calmar = ann / abs(dd) if dd != 0 else 0
    return {
        "Ann. Return": ann,
        "Ann. Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": dd,
        "Win Rate": (r > 0).mean(),
        "N Months": len(r),
    }


def var_cvar(rets: pd.Series, alpha: float = 0.05) -> dict:
    mu, sig = rets.mean(), rets.std()
    rng = np.random.default_rng(42)
    sims = rng.normal(mu, sig, 100_000)

    h_var = float(np.percentile(rets, alpha * 100))
    h_cvar = float(rets[rets <= h_var].mean())
    p_var = float(stats.norm.ppf(alpha, mu, sig))
    p_cvar = float(mu - sig * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)
    mc_var = float(np.percentile(sims, alpha * 100))
    mc_cvar = float(sims[sims <= mc_var].mean())

    return {
        "Historical": {"VaR": h_var, "CVaR": h_cvar},
        "Parametric": {"VaR": p_var, "CVaR": p_cvar},
        "Monte Carlo": {"VaR": mc_var, "CVaR": mc_cvar},
    }


def sector_annual_returns(rets_daily: pd.DataFrame) -> pd.DataFrame:
    annual_data = {}
    for sector in sorted(set(SECTORS.values())):
        tkrs = [t for t, s in SECTORS.items() if s == sector]
        annual_data[sector] = (1 + rets_daily[tkrs].mean(axis=1)).resample(ANNUAL_FREQ).prod() - 1
    df = pd.DataFrame(annual_data)
    df.index = df.index.year
    return df


def run_full_pipeline() -> dict:
    """Execute end-to-end analysis; returns all artifacts for the dashboard."""
    data = load_market_data()
    factors = build_factors(data["prices_monthly"], data["rets_daily"])

    ic_series = {
        "12-1 Momentum": compute_ic(factors["mom121"], factors["fwd_ret_1m"]),
        "6-1 Momentum": compute_ic(factors["mom6"], factors["fwd_ret_1m"]),
        "Reversal 1m": compute_ic(factors["reversal"], factors["fwd_ret_1m"]),
        "Low Volatility": compute_ic(factors["lowvol"], factors["fwd_ret_1m"]),
    }
    ic_table = pd.DataFrame([ic_summary(s, n) for n, s in ic_series.items()])

    panel = build_ml_panel(data["prices_monthly"], data["rets_daily"])
    train, val, test = split_panel(panel)

    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_val, y_val = val[FEATURE_COLS], val["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]

    model = train_xgb_model(X_train, y_train, X_val, y_val)
    metrics = model_metrics(model, X_val, y_val, X_test, y_test)

    val_test = pd.concat([val, test])
    signals = generate_ls_weights(val_test, model)
    pnl = run_backtest(signals, data["rets_monthly"])

    spy_sub = data["spy_monthly"].reindex(pnl.index)
    ew_sub = (
        data["rets_daily"].mean(axis=1)
        .resample(MONTHLY_FREQ)
        .apply(lambda x: (1 + x).prod() - 1)
        .reindex(pnl.index)
    )

    perf = pd.DataFrame({
        "ML L/S Net": performance_metrics(pnl["net"]),
        "ML L/S Gross": performance_metrics(pnl["gross"]),
        "SPY": performance_metrics(spy_sub),
        "EW Universe": performance_metrics(ew_sub),
    }).T

    port_daily = data["rets_daily"].mean(axis=1)
    var_95 = var_cvar(port_daily, 0.05)
    var_99 = var_cvar(port_daily, 0.01)

    return {
        **data,
        "factors": factors,
        "ic_series": ic_series,
        "ic_table": ic_table,
        "panel": panel,
        "train": train,
        "val": val,
        "test": test,
        "model": model,
        "metrics": metrics,
        "pnl": pnl,
        "spy_sub": spy_sub,
        "ew_sub": ew_sub,
        "perf": perf,
        "port_daily": port_daily,
        "var_95": var_95,
        "var_99": var_99,
        "sector_annual": sector_annual_returns(data["rets_daily"]),
    }
