# 📈 AI Algorithmic Trading System

ML-powered trading platform with ensemble models, SHAP explainability, risk management, and a live Streamlit dashboard.

---

## 🏗️ Architecture

```
stock_data.csv
     │
     ▼
model_pipeline.py       ← Feature engineering (38 features: RSI, MACD, BB, Volume...)
     │                     XGBoost + LightGBM Ensemble
     │                     SHAP Explainability
     ▼
trading_model.pkl       ← Serialized model artifact (deploy this)
     │
     ├──► backtester.py ← Risk-managed backtesting engine
     │
     └──► app.py        ← Streamlit dashboard (Render-deployable)
```

---

## 📦 Features

| Category | Implementation |
|---|---|
| **ML Models** | XGBoost + LightGBM soft-voting ensemble |
| **Features** | RSI (9/14/21), MACD, Bollinger Bands, OBV, Stochastic, ATR, Vol indicators |
| **Target** | Binary: 5-day forward return > 1% threshold |
| **Validation** | Time-series split (no data leakage) |
| **Explainability** | SHAP TreeExplainer on XGBoost |
| **Risk Management** | Stop-loss, max drawdown halt, position sizing |
| **Backtesting** | Sharpe, Sortino, max drawdown, win rate, profit factor |
| **Dashboard** | Streamlit + Plotly: equity curve, signals, SHAP chart |

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt

# Generate stock_data.csv (replace with real yfinance data in production)
python generate_data.py   # or use your own OHLCV CSV

# Train model → produces trading_model.pkl
python trading_system/train.py

# Run dashboard locally
streamlit run trading_system/app.py
```


**Note:** For real stock data, replace `stock_data.csv` with yfinance:
```python
import yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2024-12-31", auto_adjust=True)
df.to_csv("stock_data.csv")
```


## 📂 File Structure

```
├── trading_system/
│   ├── model_pipeline.py   # Feature engineering + model training
│   ├── backtester.py       # Strategy simulation + risk engine
│   ├── app.py              # Streamlit dashboard
│   ├── train.py            # One-time training script → .pkl
│   ├── requirements.txt    # Render-compatible deps
│   └── trading_model.pkl   # Trained model artifact (1.5MB)
├── stock_data.csv          # OHLCV input data
└── README.md
```

