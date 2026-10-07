"""
Portfolio optimizer demo (textbook methods only, long-only).
Educational use only. Contains no proprietary methodology.
"""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

# ---------------------------------------------------------------
# Step 0: all tunable parameters live here (nothing hard-coded below)
# ---------------------------------------------------------------
CONFIG = {
    "page_title": "Portfolio Optimizer Demo",
    "default_tickers": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM",
        "JNJ", "XOM", "PG", "KO", "PEP", "WMT", "HD", "UNH", "V", "MA",
        "CVX", "MRK", "ABBV", "COST", "MCD", "CSCO", "ORCL",
    ],
    "yahoo_market": "^GSPC",          # market index used for beta (Yahoo path)
    "default_start_years_back": 5,
    "min_assets": 2,
    "trading_days": 252,              # periods per year for daily Yahoo data
    "freq_map": [(4, 252), (10, 52), (40, 12), (120, 4)],  # (max median gap in days, periods per year)
    "fallback_ann_factor": 1,
    "default_rf_pct": 4.64,
    "default_theta": 3.0,
    "theta_min": 0.5,
    "theta_max": 15.0,
    "default_target_return_pct": 3.0,
    "default_investment": 1_000_000,
    "default_max_weight_pct": 40,
    "frontier_points": 25,
    "frontier_max_assets": 40,        # frontier is pre-ticked only up to this size
    "ridge": 1e-8,                    # tiny diagonal add-on keeps the covariance matrix invertible
    "max_iter": 500,
    "min_display_weight": 0.0005,
    "list_height": 320,
    "optimizers": {                   # label shown in the app -> internal key
        "Max Sharpe ratio": "sharpe",
        "Max utility (risk aversion θ)": "utility",
        "Min variance": "minvar",
        "Min variance for a target return": "target",
        "Max Treynor ratio (needs market index)": "treynor",
        "Max Sortino ratio": "sortino",
        "Risk parity (equal risk contribution)": "riskparity",
        "Max diversification ratio": "diversification",
    },
}

st.set_page_config(page_title=CONFIG["page_title"], layout="wide")


# Step 1: download adjusted close prices (cached so repeated clicks are fast)
@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    data = yf.download(list(tickers), start=start, end=end,
                       auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=list(tickers)[0])
    # drop tickers with no data, then rows with gaps
    return data.dropna(axis=1, how="all").dropna()


# Step 2: portfolio statistics from weights
def portfolio_stats(w, mu, cov, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


# Step 3: maximize the Sharpe ratio (long-only, weights sum to 1, weight cap)
def max_sharpe(mu, cov, rf, max_w):
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, max_w)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(lambda w: -portfolio_stats(w, mu, cov, rf)[2], x0,
                   method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


# Step 4: efficient frontier = minimum variance for each target return
def efficient_frontier(mu, cov, max_w, n_points):
    n = len(mu)
    bounds = [(0.0, max_w)] * n
    targets = np.linspace(mu.min(), mu.max(), n_points)
    pts = []
    for t in targets:
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=t: w @ mu - t}]
        r = minimize(lambda w: w @ cov @ w, np.full(n, 1.0 / n),
                     method="SLSQP", bounds=bounds, constraints=cons)
        if r.success:
            pts.append((float(np.sqrt(r.x @ cov @ r.x)), t))
    return pts


# Step 5: read an uploaded CSV or Excel file (first column = date, one column per asset)
def read_uploaded(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file, index_col=0)
    else:
        # sep=None lets pandas detect comma or semicolon separators
        raw = pd.read_csv(file, index_col=0, sep=None, engine="python")
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw[~raw.index.isna()].sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw.apply(pd.to_numeric, errors="coerce")
    return raw.dropna(axis=1, how="all").dropna(how="all")


# Step 6: guess periods per year (252 / 52 / 12 / 4 / 1) from the date spacing
def detect_ann_factor(index):
    gaps = pd.Series(index).diff().dt.days.dropna()
    if gaps.empty:
        return CONFIG["fallback_ann_factor"]
    median_gap = gaps.median()
    for max_gap, factor in CONFIG["freq_map"]:
        if median_gap <= max_gap:
            return factor
    return CONFIG["fallback_ann_factor"]


# Step 7: Sortino ratio with the risk-free rate per period as minimum acceptable return
def sortino_ratio(w, R, rf, af):
    r = R @ w
    mar = rf / af
    downside_vol = np.sqrt(np.mean(np.minimum(r - mar, 0.0) ** 2) * af)
    ret = r.mean() * af
    return (ret - rf) / max(downside_vol, 1e-9)


# Step 8: generic long-only solver (weights sum to 1, each weight between 0 and the cap)
def solve(obj, n, max_w, extra_cons=None):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}] + (extra_cons or [])
    res = minimize(obj, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, max_w)] * n, constraints=cons,
                   options={"maxiter": CONFIG["max_iter"], "ftol": 1e-10})
    return res.x, bool(res.success)


# Step 9: choose the objective function by optimizer key and solve
def optimize_portfolio(kind, mu, cov, rf, max_w, theta, target, beta, R, af):
    n = len(mu)
    vols = np.sqrt(np.diag(cov))
    if kind == "sharpe":
        return max_sharpe(mu, cov, rf, max_w), True
    if kind == "utility":
        # mean-variance utility U = mu - (theta / 2) * variance
        return solve(lambda w: -(w @ mu - 0.5 * theta * (w @ cov @ w)), n, max_w)
    if kind == "minvar":
        return solve(lambda w: w @ cov @ w, n, max_w)
    if kind == "target":
        cons = [{"type": "ineq", "fun": lambda w: w @ mu - target}]
        return solve(lambda w: w @ cov @ w, n, max_w, cons)
    if kind == "treynor":
        cons = [{"type": "ineq", "fun": lambda w: w @ beta - 1e-3}]
        return solve(lambda w: -(w @ mu - rf) / (w @ beta), n, max_w, cons)
    if kind == "sortino":
        return solve(lambda w: -sortino_ratio(w, R, rf, af), n, max_w)
    if kind == "riskparity":
        def rp_obj(w):
            var = w @ cov @ w
            contrib = w * (cov @ w) / var   # share of total variance per asset
            return np.sum((contrib - 1.0 / n) ** 2)
        return solve(rp_obj, n, max_w)
    if kind == "diversification":
        return solve(lambda w: -(w @ vols) / np.sqrt(w @ cov @ w), n, max_w)
    raise ValueError(f"Unknown optimizer: {kind}")


# Step 10: report for the chosen portfolio
def portfolio_report(w, mu, cov, rf, theta, R, af, beta):
    ret, vol, sharpe = portfolio_stats(w, mu, cov, rf)
    vols = np.sqrt(np.diag(cov))
    r = R @ w
    mar = rf / af
    out = {
        "ret": ret, "vol": vol, "sharpe": sharpe,
        "utility": ret - 0.5 * theta * vol ** 2,
        "sortino": sortino_ratio(w, R, rf, af),
        "downside_vol": float(np.sqrt(np.mean(np.minimum(r - mar, 0.0) ** 2) * af)),
        "div_ratio": float((w @ vols) / vol) if vol > 0 else np.nan,
        "max_w": float(w.max()),
    }
    if beta is not None:
        bp = float(w @ beta)
        out["beta"] = bp
        out["treynor"] = (ret - rf) / bp if bp > 0 else np.nan
    return out


# Step 11: vertical asset list with checkboxes, search filter and select all / clear
def asset_selector(cols):
    universe = tuple(cols)
    if st.session_state.get("sel_universe") != universe:
        st.session_state["sel_universe"] = universe
        st.session_state["sel"] = set(cols)
        st.session_state["sel_ver"] = 0

    c1, c2, c3 = st.columns([3, 1, 1])
    flt = c1.text_input("Filter list", "", placeholder="type to search assets")
    visible = [c for c in cols if flt.lower() in str(c).lower()]
    if c2.button("Select all shown"):
        st.session_state["sel"] |= set(visible)
        st.session_state["sel_ver"] += 1
    if c3.button("Clear all shown"):
        st.session_state["sel"] -= set(visible)
        st.session_state["sel_ver"] += 1

    table = pd.DataFrame({"Select": [c in st.session_state["sel"] for c in visible],
                          "Asset": visible})
    edited = st.data_editor(
        table, hide_index=True, disabled=["Asset"], height=CONFIG["list_height"],
        width="stretch", key=f"sel_editor_{st.session_state['sel_ver']}_{flt}")
    for asset, flag in zip(edited["Asset"], edited["Select"]):
        if flag:
            st.session_state["sel"].add(asset)
        else:
            st.session_state["sel"].discard(asset)
    chosen = [c for c in cols if c in st.session_state["sel"]]
    st.caption(f"{len(chosen)} of {len(cols)} assets selected")
    return chosen


# ===== UI =====
# ---------------------------------------------------------------
# Home page: data source
# ---------------------------------------------------------------
st.title("Portfolio Optimizer Demo")
st.caption("Teaching demo: classic textbook optimizers on historical data. "
           "Not investment advice.")

source = st.radio("Data source", ["Upload file (CSV / Excel)", "Yahoo Finance (demo)"],
                  horizontal=True)
market = None

if source.startswith("Upload"):
    up = st.file_uploader(
        "Upload prices or returns (first column = date, one column per asset)",
        type=["csv", "xlsx", "xls"])
    if up is None:
        st.info("Upload a file to continue. The asset list and the date range "
                "appear after the upload.")
        st.stop()
    data = read_uploaded(up)
    if data.shape[1] < CONFIG["min_assets"] or len(data) < 3:
        st.error("The file needs a date column, at least two asset columns "
                 "and at least three rows.")
        st.stop()
    kind_values = st.radio("The file contains", ["Prices", "Returns (decimal)"],
                           horizontal=True)
    d_min, d_max = data.index.min().date(), data.index.max().date()
    ann_default = detect_ann_factor(data.index)
    st.caption(f"{data.shape[1]} columns, {len(data)} rows, {d_min} to {d_max}, "
               f"detected {ann_default} periods per year.")

    # date range is shown after the upload and limited to the data
    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", d_min, min_value=d_min, max_value=d_max)
    end = col2.date_input("End date", d_max, min_value=d_min, max_value=d_max)

    mk = st.selectbox("Market index for beta / Treynor (optional)",
                      ["None"] + list(data.columns))
    market = None if mk == "None" else mk

    st.subheader("Select assets")
    chosen = asset_selector([c for c in data.columns if c != market])
else:
    data = None
    kind_values = "Prices"
    ann_default = CONFIG["trading_days"]
    col1, col2 = st.columns(2)
    with col1:
        chosen = st.multiselect("Select assets", CONFIG["default_tickers"],
                                default=CONFIG["default_tickers"][:8])
        extra = st.text_input("Add other tickers (comma-separated, optional)")
        chosen = list(dict.fromkeys(
            chosen + [t.strip().upper() for t in extra.split(",") if t.strip()]))
    with col2:
        today = dt.date.today()
        start = st.date_input(
            "Start date",
            today - dt.timedelta(days=365 * CONFIG["default_start_years_back"]))
        end = st.date_input("End date", today)
    if st.checkbox(f"Use {CONFIG['yahoo_market']} as market index (beta / Treynor)", True):
        market = CONFIG["yahoo_market"]

# ---------------------------------------------------------------
# Optimizer settings
# ---------------------------------------------------------------
st.subheader("Optimizer settings")
s1, s2, s3 = st.columns(3)
opt_label = s1.selectbox("Optimizer", list(CONFIG["optimizers"]))
opt_kind = CONFIG["optimizers"][opt_label]
rf_pct = s2.number_input("Risk-free rate (% p.a.)", 0.0, 20.0,
                         CONFIG["default_rf_pct"], 0.01)
theta = s3.number_input("Risk aversion θ (used by utility)", CONFIG["theta_min"],
                        CONFIG["theta_max"], CONFIG["default_theta"], 0.5)

s4, s5, s6 = st.columns(3)
cap_pct = s4.slider("Maximum weight per asset (%)", 1, 100,
                    CONFIG["default_max_weight_pct"])
investment = s5.number_input("Investment amount", 1000, 10**12,
                             CONFIG["default_investment"], 1000)
ann_factor = int(s6.number_input("Periods per year", 1, 365, int(ann_default), 1))

target_pct = CONFIG["default_target_return_pct"]
if opt_kind == "target":
    target_pct = st.number_input("Target return (% p.a.)", -50.0, 200.0,
                                 CONFIG["default_target_return_pct"], 0.5)
show_frontier = st.checkbox("Show efficient frontier (slower with many assets)",
                            value=len(chosen) <= CONFIG["frontier_max_assets"])

# Step 12: run only when the button is pressed
if st.button("Optimize", type="primary"):
    if len(chosen) < CONFIG["min_assets"]:
        st.error(f"Select at least {CONFIG['min_assets']} assets.")
        st.stop()
    if start >= end:
        st.error("Start date must be before end date.")
        st.stop()
    if opt_kind == "treynor" and market is None:
        st.error("The Treynor optimizer needs a market index. Select one above.")
        st.stop()

    with st.spinner("Preparing data and optimizing..."):
        needed = list(dict.fromkeys(chosen + ([market] if market else [])))
        if data is not None:
            window = data.loc[pd.Timestamp(start):pd.Timestamp(end), needed]
        else:
            window = load_prices(tuple(needed), start, end)
            chosen = [c for c in chosen if c in window.columns]
            if market not in window.columns:
                market = None
        if len(window) < 3:
            st.error("Not enough data in this date range.")
            st.stop()

        # period returns from prices, or use the uploaded returns as they are
        if kind_values == "Prices":
            rets = window.pct_change(fill_method=None)
        else:
            rets = window.copy()
        rets = rets.dropna(how="all")
        if market:
            rets = rets.dropna(subset=[market])

        # assets with gaps in the chosen window are dropped and reported
        dropped = [c for c in chosen if rets[c].isna().any()]
        names = [c for c in chosen if c not in dropped]
        if dropped:
            st.warning(f"{len(dropped)} asset(s) dropped because of missing values "
                       f"in this window: {', '.join(map(str, dropped[:15]))}"
                       f"{' ...' if len(dropped) > 15 else ''}")
        if len(names) < CONFIG["min_assets"]:
            st.error("Fewer than two assets with complete data remain.")
            st.stop()

        rf, max_w = rf_pct / 100.0, cap_pct / 100.0
        if max_w * len(names) < 1.0:
            st.error("Weight cap too low for this number of assets. "
                     "Raise the cap or select more assets.")
            st.stop()
        if len(rets) < len(names):
            st.warning("Fewer observations than assets: the covariance matrix is "
                       "singular and results can be unstable.")

        # annualized mean returns and covariance (v1.0 logic, with periods per year)
        R = rets[names].values
        mu = R.mean(axis=0) * ann_factor
        cov = np.cov(R, rowvar=False) * ann_factor + np.eye(len(names)) * CONFIG["ridge"]

        # beta of each asset against the market index (if one is selected)
        beta = None
        if market:
            rm = rets[market].values
            beta = ((R - R.mean(axis=0)).T @ (rm - rm.mean())) / \
                   ((len(rm) - 1) * np.var(rm, ddof=1))

        w, ok = optimize_portfolio(opt_kind, mu, cov, rf, max_w, theta,
                                   target_pct / 100.0, beta, R, ann_factor)
        if abs(w.sum() - 1.0) > 1e-4:
            st.error("The optimizer did not return a valid solution. "
                     "For the target-return optimizer, the target may be "
                     "unreachable under the weight cap.")
            st.stop()
        if not ok:
            st.warning("The solver reported incomplete convergence "
                       "(common for Sortino with few observations). "
                       "Treat the weights as approximate.")
        w = np.clip(w, 0.0, None)
        w = w / w.sum()
        rep = portfolio_report(w, mu, cov, rf, theta, R, ann_factor, beta)
        frontier = (efficient_frontier(mu, cov, max_w, CONFIG["frontier_points"])
                    if show_frontier else [])

    # Step 13: results
    st.subheader(f"Result: {opt_label}")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Expected return (p.a.)", f"{rep['ret']:.1%}")
    a2.metric("Volatility (p.a.)", f"{rep['vol']:.1%}")
    a3.metric("Sharpe ratio", f"{rep['sharpe']:.2f}")
    a4.metric(f"Utility (θ = {theta:g})", f"{rep['utility']:.1%}")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Sortino ratio", f"{rep['sortino']:.2f}")
    b2.metric("Downside volatility", f"{rep['downside_vol']:.1%}")
    b3.metric("Diversification ratio", f"{rep['div_ratio']:.2f}")
    b4.metric("Largest weight", f"{rep['max_w']:.1%}")
    if beta is not None:
        c1m, c2m, _, _ = st.columns(4)
        c1m.metric("Portfolio beta", f"{rep['beta']:.2f}")
        c2m.metric("Treynor ratio", f"{rep['treynor']:.2f}")

    vols = np.sqrt(np.diag(cov))
    table = pd.DataFrame({"Asset": names, "Weight": w,
                          "Amount": w * investment,
                          "Asset Sharpe": (mu - rf) / vols})
    if beta is not None:
        table["Asset beta"] = beta
        table["Asset Treynor"] = np.where(beta > 0, (mu - rf) / np.where(beta > 0, beta, 1), np.nan)
    table = table[table["Weight"] > CONFIG["min_display_weight"]] \
        .sort_values("Weight", ascending=False)
    fmt = {"Weight": "{:.1%}", "Amount": "{:,.0f}", "Asset Sharpe": "{:.2f}",
           "Asset beta": "{:.2f}", "Asset Treynor": "{:.2f}"}

    left, right = st.columns(2)
    with left:
        st.subheader("Optimal weights")
        st.dataframe(table.style.format({k: v for k, v in fmt.items() if k in table}),
                     hide_index=True, width="stretch")
    with right:
        st.subheader("Risk and return")
        fig, ax = plt.subplots()
        if frontier:
            ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                    label="Efficient frontier")
        ax.scatter(vols, mu, s=15, color="grey", label="Single assets")
        ax.scatter([rep["vol"]], [rep["ret"]], color="red", zorder=5, label="Portfolio")
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Expected return")
        ax.legend()
        st.pyplot(fig)

    st.info("Notes: results are based on historical data and are in-sample, "
            "so they say nothing reliable about future performance. "
            "Utility is U = expected return - (θ / 2) x variance. "
            "Using today's index constituents introduces survivorship bias.")

# ---------------------------------------------------------------
# Version log
# v1.0 - Initial version: asset and date selection, Optimize button,
#        max-Sharpe weights (long-only, weight cap), efficient frontier,
#        all parameters in CONFIG.
# v1.1 - Added: CSV/Excel upload (prices or returns, 30-100+ assets),
#        vertical asset list with checkboxes, search filter and select all/clear,
#        start/end date fields shown after upload and limited to the data range,
#        automatic periods-per-year detection (editable),
#        optional market index for beta and Treynor,
#        optimizers: max utility (risk aversion θ), min variance,
#        min variance for target return, max Treynor, max Sortino,
#        risk parity, max diversification ratio (max Sharpe kept unchanged),
#        investment amount with per-asset Sharpe/beta/Treynor table,
#        extra result metrics (utility, Sortino, downside volatility,
#        diversification ratio, beta, Treynor), checkbox for the frontier,
#        convergence warning instead of a hard stop when the solver is not fully
#        converged but the weights are valid,
#        CONFIG extended; v1.0 functions portfolio_stats, max_sharpe,
#        efficient_frontier and load_prices unchanged.
# ---------------------------------------------------------------
