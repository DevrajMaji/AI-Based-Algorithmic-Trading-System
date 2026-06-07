"""
AI-Based Algorithmic Trading System
Model Pipeline: Feature Engineering → XGBoost + LightGBM Ensemble → SHAP → model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load OHLCV data. Expects columns: Open, High, Low, Close, Volume."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df.sort_index()
    print(f"[DATA] Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    bandwidth = (upper - lower) / (sma + 1e-9)
    return upper, sma, lower, pct_b, bandwidth


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 40+ technical indicator features."""
    f = df.copy()
    c = f["Close"]
    h = f["High"]
    l = f["Low"]
    v = f["Volume"]

    # ── Returns ──
    for n in [1, 3, 5, 10, 20]:
        f[f"ret_{n}d"] = c.pct_change(n)

    # ── RSI (multiple periods) ──
    for period in [9, 14, 21]:
        f[f"rsi_{period}"] = compute_rsi(c, period)

    # ── MACD ──
    macd, signal, hist = compute_macd(c)
    f["macd"] = macd
    f["macd_signal"] = signal
    f["macd_hist"] = hist
    f["macd_cross"] = (macd > signal).astype(int)

    # ── Bollinger Bands ──
    bb_upper, bb_mid, bb_lower, pct_b, bandwidth = compute_bollinger_bands(c)
    f["bb_upper"] = bb_upper
    f["bb_lower"] = bb_lower
    f["bb_pct_b"] = pct_b
    f["bb_bandwidth"] = bandwidth
    f["bb_squeeze"] = (bandwidth < bandwidth.rolling(20).mean()).astype(int)

    # ── Moving Averages & Crossovers ──
    for w in [5, 10, 20, 50, 200]:
        f[f"sma_{w}"] = c.rolling(w).mean()
        f[f"price_vs_sma{w}"] = (c - f[f"sma_{w}"]) / f[f"sma_{w}"]

    f["golden_cross"] = (f["sma_50"] > f["sma_200"]).astype(int)
    f["sma5_vs_20"] = (f["sma_5"] > f["sma_20"]).astype(int)

    # ── Volume Indicators ──
    f["vol_sma20"] = v.rolling(20).mean()
    f["vol_ratio"] = v / (f["vol_sma20"] + 1)
    f["obv"] = (np.sign(c.diff()) * v).cumsum()
    f["obv_slope"] = f["obv"].diff(5)

    # ── Volatility ──
    f["atr"] = (h - l).rolling(14).mean()
    f["hist_vol_10"] = c.pct_change().rolling(10).std() * np.sqrt(252)
    f["hist_vol_20"] = c.pct_change().rolling(20).std() * np.sqrt(252)

    # ── Momentum ──
    f["roc_5"] = (c / c.shift(5) - 1) * 100
    f["roc_10"] = (c / c.shift(10) - 1) * 100
    f["momentum_10"] = c - c.shift(10)

    # ── Candlestick patterns ──
    f["body_size"] = (c - df["Open"]).abs() / (h - l + 1e-9)
    f["upper_shadow"] = (h - c.clip(upper=df["Open"])) / (h - l + 1e-9)
    f["lower_shadow"] = (c.clip(lower=df["Open"]) - l) / (h - l + 1e-9)
    f["is_bullish_candle"] = (c > df["Open"]).astype(int)

    # ── Stochastic Oscillator ──
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    f["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()

    # Drop raw price columns used only for computation
    drop_cols = [col for col in f.columns if col.startswith("sma_") or col.startswith("bb_upper") or col.startswith("bb_lower")]
    f.drop(columns=drop_cols, inplace=True, errors="ignore")

    return f


# ─────────────────────────────────────────────
# 3. TARGET LABEL
# ─────────────────────────────────────────────

def create_target(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.01) -> pd.Series:
    """
    Binary target: 1 if stock returns > threshold% over next N days, else 0.
    This avoids look-ahead bias — target is shifted backward after calculation.
    """
    fwd_return = df["Close"].shift(-forward_days) / df["Close"] - 1
    target = (fwd_return > threshold).astype(int)
    return target


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────

def build_ensemble():
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
        weights=[0.55, 0.45],
    )
    return ensemble


def train_model(df: pd.DataFrame):
    print("\n[PIPELINE] Engineering features...")
    features_df = engineer_features(df)
    target = create_target(df)

    # Align and drop NaN rows
    data = features_df.join(target.rename("target"), how="left")
    data.dropna(inplace=True)
    data = data[data["target"].notna()]

    feature_cols = [c for c in data.columns if c not in ["Open", "High", "Low", "Close", "Volume", "target"]]
    X = data[feature_cols]
    y = data["target"]

    print(f"[PIPELINE] Features: {len(feature_cols)} | Samples: {len(X)} | Label balance: {y.mean():.2%} positive")

    # Time-series aware split (no shuffle!)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("[PIPELINE] Training XGBoost + LightGBM ensemble...")
    model = build_ensemble()
    model.fit(X_train_s, y_train)

    # Evaluate
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\n[EVAL] ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["SELL/HOLD", "BUY"]))

    # SHAP values (on XGB estimator)
    print("[SHAP] Computing feature importance...")
    import shap
    xgb_model = model.estimators_[0]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_s[:200])  # subset for speed

    shap_importance = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)

    print("\n[SHAP] Top 10 Features:")
    print(shap_importance.head(10).to_string(index=False))

    # Package everything
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "shap_importance": shap_importance,
        "metrics": {
            "roc_auc": round(auc, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "test_data": {
            "X_test": X_test,
            "y_test": y_test,
            "proba": proba,
            "preds": preds,
        },
        "all_data": data,
    }

    # Save .pkl
    joblib.dump(artifact, "trading_model.pkl")
    print("\n[SAVED] trading_model.pkl ✓")
    return artifact


if __name__ == "__main__":
    df = load_data("stock_data.csv")
    artifact = train_model(df)
    print("\n[DONE] Model pipeline complete.")
