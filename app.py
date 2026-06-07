"""
AI Algorithmic Trading Dashboard
Streamlit app — deployable on Render
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys

# ── path fix for imports ──
sys.path.insert(0, os.path.dirname(__file__))
from model_pipeline import load_data, engineer_features, create_target, train_model
from backtester import run_backtest_from_pkl, benchmark_buy_hold, Backtester

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AlgoTrader AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Space Mono', monospace; background: #0a0e1a; color: #e0e6f0; }
  h1, h2, h3 { font-family: 'Syne', sans-serif; }
  .metric-card { background: #111827; border: 1px solid #1f2d45; border-radius: 8px; padding: 16px; text-align: center; }
  .metric-value { font-size: 1.8rem; font-weight: 700; }
  .metric-label { font-size: 0.75rem; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase; }
  .signal-buy { color: #10b981; font-weight: 700; }
  .signal-sell { color: #ef4444; font-weight: 700; }
  .signal-hold { color: #f59e0b; font-weight: 700; }
  .stButton>button { background: #1d4ed8; color: white; border: none; border-radius: 6px; font-family: 'Space Mono'; }
  div[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    initial_capital = st.number_input("Initial Capital ($)", value=100_000, step=10_000)
    signal_threshold = st.slider("Signal Confidence Threshold", 0.40, 0.80, 0.55, 0.01)
    forward_days = st.selectbox("Prediction Horizon (days)", [3, 5, 10], index=1)
    retrain = st.button("🚀 Train & Run System")
    st.markdown("---")
    st.markdown("**Risk Parameters**")
    stop_loss = st.slider("Stop Loss (%)", 1, 15, 5)
    max_dd = st.slider("Max Drawdown Halt (%)", 5, 30, 15)
    pos_size = st.slider("Position Size (%)", 5, 50, 10)
    st.markdown("---")
    st.caption("AlgoTrader AI · Built with XGBoost + LightGBM · SHAP Explainability")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("# 📈 AlgoTrader AI")
st.markdown("*Ensemble ML Trading System with Risk Management & Explainability*")
st.markdown("---")


# ─────────────────────────────────────────────
# LOAD / TRAIN
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Training model pipeline...")
def load_or_train(data_path, fwd_days):
    df = load_data(data_path)
    artifact = train_model(df)
    return artifact

PKL = "trading_model.pkl"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "stock_data.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "stock_data.csv"

artifact = None
bt_result = None

if retrain or (os.path.exists(PKL)):
    try:
        if retrain:
            st.cache_resource.clear()
        with st.spinner("Loading model & running backtest..."):
            artifact = load_or_train(DATA_PATH, forward_days)
            from backtester import run_backtest_from_pkl, Backtester
            data = artifact["all_data"]
            test_data = artifact["test_data"]
            X_test = test_data["X_test"]
            prices = data.loc[X_test.index, "Close"]
            signals = pd.Series(test_data["preds"], index=X_test.index)
            probas = pd.Series(test_data["proba"], index=X_test.index)

            from backtester import Backtester, benchmark_buy_hold, RiskManager
            bt = Backtester(initial_capital=initial_capital, transaction_cost=0.001)
            bt.risk_manager.stop_loss_pct = stop_loss / 100
            bt.risk_manager.max_drawdown_pct = max_dd / 100
            bt.risk_manager.position_size_pct = pos_size / 100
            bt_result = bt.run(prices, signals, probas, threshold=signal_threshold)
            bt_result["benchmark"] = benchmark_buy_hold(prices, initial_capital)
            bt_result["prices"] = prices
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.info("👈 Click **Train & Run System** in the sidebar to start.")
    st.stop()


# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────

metrics = bt_result["metrics"]
m = metrics

col1, col2, col3, col4, col5, col6 = st.columns(6)
def metric_card(col, label, value, color="#10b981"):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color}">{value}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

ret_color = "#10b981" if m["Total Return (%)"] > 0 else "#ef4444"
metric_card(col1, "Total Return", f"{m['Total Return (%)']:+.1f}%", ret_color)
metric_card(col2, "Sharpe Ratio", f"{m['Sharpe Ratio']:.3f}", "#3b82f6")
metric_card(col3, "Max Drawdown", f"{m['Max Drawdown (%)']:.1f}%", "#f59e0b")
metric_card(col4, "Win Rate", f"{m['Win Rate (%)']:.1f}%", "#8b5cf6")
metric_card(col5, "Total Trades", f"{m['Total Trades']}", "#06b6d4")
metric_card(col6, "ROC-AUC", f"{artifact['metrics']['roc_auc']:.4f}", "#10b981")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📊 Equity Curve", "🔔 Trade Signals", "🧠 SHAP Explainability", "📋 Full Metrics"])

# ── TAB 1: Equity Curve ──
with tab1:
    eq = bt_result["equity_curve"]
    bh = bt_result["benchmark"]
    prices = bt_result["prices"]
    trades_df = bt_result["trades"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("Portfolio vs Buy & Hold", "Price + Buy/Sell Signals", "Daily Returns")
    )

    # Equity curves
    fig.add_trace(go.Scatter(x=eq.index, y=eq["portfolio"], name="ML Strategy",
                             line=dict(color="#3b82f6", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh.index, y=bh["portfolio"], name="Buy & Hold",
                             line=dict(color="#64748b", width=1.5, dash="dot")), row=1, col=1)

    # Price
    fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Price",
                             line=dict(color="#94a3b8", width=1)), row=2, col=1)

    # Buy/sell markers
    if not trades_df.empty:
        buys = trades_df[trades_df["type"] == "BUY"]
        sells = trades_df[trades_df["type"].isin(["SELL", "STOP_LOSS"])]
        stops = trades_df[trades_df["type"] == "STOP_LOSS"]
        if len(buys):
            fig.add_trace(go.Scatter(x=buys["date"], y=buys["price"],
                                     mode="markers", name="BUY",
                                     marker=dict(color="#10b981", size=10, symbol="triangle-up")), row=2, col=1)
        if len(sells):
            fig.add_trace(go.Scatter(x=sells["date"], y=sells["price"],
                                     mode="markers", name="SELL",
                                     marker=dict(color="#ef4444", size=10, symbol="triangle-down")), row=2, col=1)

    # Daily returns
    daily_ret = eq["portfolio"].pct_change().dropna() * 100
    colors_ret = ["#10b981" if r > 0 else "#ef4444" for r in daily_ret]
    fig.add_trace(go.Bar(x=daily_ret.index, y=daily_ret, name="Daily Return %",
                         marker_color=colors_ret, showlegend=False), row=3, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(color="#e0e6f0", family="Space Mono"),
        legend=dict(bgcolor="#111827", bordercolor="#1f2d45"),
        xaxis3=dict(gridcolor="#1f2d45"),
        yaxis=dict(gridcolor="#1f2d45"),
        yaxis2=dict(gridcolor="#1f2d45"),
        yaxis3=dict(gridcolor="#1f2d45"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 2: Live Signal Dashboard ──
with tab2:
    st.markdown("### 🔔 Latest ML Signals")

    data = artifact["all_data"]
    X_test = artifact["test_data"]["X_test"]
    proba_series = pd.Series(artifact["test_data"]["proba"], index=X_test.index)
    pred_series = pd.Series(artifact["test_data"]["preds"], index=X_test.index)
    price_series = data.loc[X_test.index, "Close"]

    # Latest 30 signals
    latest = pd.DataFrame({
        "Date": X_test.index[-30:],
        "Price": price_series[-30:].values,
        "Signal": pred_series[-30:].values,
        "Confidence": proba_series[-30:].values,
    }).sort_values("Date", ascending=False)

    def signal_label(s, c):
        if s == 1 and c > signal_threshold:
            return "🟢 BUY"
        elif s == 0:
            return "🔴 SELL/HOLD"
        return "🟡 WEAK BUY"

    latest["Action"] = [signal_label(s, c) for s, c in zip(latest["Signal"], latest["Confidence"])]
    latest["Confidence"] = (latest["Confidence"] * 100).round(1).astype(str) + "%"
    latest["Price"] = latest["Price"].round(2)

    # Current signal highlight
    last = latest.iloc[0]
    sig_color = "#10b981" if "BUY" in last["Action"] and "WEAK" not in last["Action"] else "#ef4444"
    st.markdown(f"""
    <div style="background:#111827;border:2px solid {sig_color};border-radius:10px;padding:20px;margin-bottom:20px;text-align:center">
      <div style="font-family:Syne;font-size:2rem;color:{sig_color}">{last['Action']}</div>
      <div style="color:#94a3b8;margin-top:8px">Latest signal as of {str(last['Date'])[:10]} · Price: ${last['Price']} · Confidence: {last['Confidence']}</div>
    </div>""", unsafe_allow_html=True)

    st.dataframe(latest.reset_index(drop=True), use_container_width=True,
                 column_config={"Confidence": st.column_config.TextColumn()})

    # Probability distribution
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=proba_series, nbinsx=40,
        marker_color="#3b82f6",
        opacity=0.8, name="Signal Probability"
    ))
    fig2.add_vline(x=signal_threshold, line_dash="dash", line_color="#f59e0b",
                   annotation_text=f"Threshold: {signal_threshold}", annotation_position="top right")
    fig2.update_layout(
        title="Signal Confidence Distribution",
        xaxis_title="Buy Probability",
        yaxis_title="Frequency",
        height=350,
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(color="#e0e6f0", family="Space Mono"),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── TAB 3: SHAP ──
with tab3:
    st.markdown("### 🧠 SHAP Feature Importance (Explainable AI)")
    st.caption("SHAP (SHapley Additive exPlanations) reveals *why* the model makes each prediction.")

    shap_df = artifact["shap_importance"].head(20)

    fig3 = go.Figure(go.Bar(
        x=shap_df["shap_importance"],
        y=shap_df["feature"],
        orientation="h",
        marker=dict(
            color=shap_df["shap_importance"],
            colorscale="Blues",
            showscale=False,
        ),
    ))
    fig3.update_layout(
        title="Top 20 Features by Mean |SHAP| Value",
        xaxis_title="Mean |SHAP| Importance",
        height=550,
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(color="#e0e6f0", family="Space Mono"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top 10 Predictive Features**")
        st.dataframe(shap_df.head(10).reset_index(drop=True), use_container_width=True)
    with col_b:
        st.markdown("**Feature Category Breakdown**")
        cats = {
            "RSI": shap_df[shap_df["feature"].str.startswith("rsi")]["shap_importance"].sum(),
            "MACD": shap_df[shap_df["feature"].str.startswith("macd")]["shap_importance"].sum(),
            "Bollinger": shap_df[shap_df["feature"].str.startswith("bb")]["shap_importance"].sum(),
            "Returns": shap_df[shap_df["feature"].str.startswith("ret")]["shap_importance"].sum(),
            "Volume": shap_df[shap_df["feature"].str.startswith("vol") | shap_df["feature"].str.startswith("obv")]["shap_importance"].sum(),
            "MA": shap_df[shap_df["feature"].str.startswith("price_vs") | shap_df["feature"].str.startswith("sma")]["shap_importance"].sum(),
            "Other": 0,
        }
        cat_df = pd.DataFrame(list(cats.items()), columns=["Category", "Importance"])
        fig4 = px.pie(cat_df, names="Category", values="Importance",
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig4.update_layout(paper_bgcolor="#0a0e1a", font=dict(color="#e0e6f0"), height=280)
        st.plotly_chart(fig4, use_container_width=True)


# ── TAB 4: Full Metrics ──
with tab4:
    st.markdown("### 📋 Full Performance Report")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Strategy Metrics**")
        met_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        st.dataframe(met_df, use_container_width=True)

    with col_m2:
        st.markdown("**Model Metrics**")
        model_met = {
            "ROC-AUC Score": artifact["metrics"]["roc_auc"],
            "Train Samples": artifact["metrics"]["train_size"],
            "Test Samples": artifact["metrics"]["test_size"],
            "Features Used": len(artifact["feature_cols"]),
            "Ensemble": "XGBoost (55%) + LightGBM (45%)",
            "Prediction Horizon": f"{forward_days} days",
        }
        st.dataframe(pd.DataFrame(list(model_met.items()), columns=["Metric", "Value"]),
                     use_container_width=True)

    if not bt_result["trades"].empty:
        st.markdown("**Trade Log**")
        td = bt_result["trades"].copy()
        td["pnl"] = td["pnl"].round(2)
        td["price"] = td["price"].round(2)
        st.dataframe(td, use_container_width=True)
