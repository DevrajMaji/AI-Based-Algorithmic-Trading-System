"""
train.py — Run this ONCE to generate trading_model.pkl for Render deployment.

Usage:
    python train.py

Output:
    trading_model.pkl  (deploy this alongside app.py on Render)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from model_pipeline import load_data, train_model
import shutil

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "stock_data.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "stock_data.csv"

print("=" * 55)
print("  AI Algorithmic Trading System — Model Training")
print("=" * 55)

df = load_data(DATA_PATH)
artifact = train_model(df)

# Copy pkl next to app.py for Render
dest = os.path.join(os.path.dirname(__file__), "trading_model.pkl")
import joblib
joblib.dump(artifact, dest)
print(f"\n✅ Saved: {dest}")
print("   Deploy this file with app.py on Render.")
print("\nRender start command:")
print("  streamlit run trading_system/app.py --server.port $PORT --server.address 0.0.0.0")
