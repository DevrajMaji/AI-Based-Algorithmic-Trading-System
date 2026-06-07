"""
Backtesting Engine
Simulates trading strategy using ML signals with risk management.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────

class RiskManager:
    def __init__(self, max_drawdown_pct=0.15, position_size_pct=0.1, stop_loss_pct=0.05):
        self.max_drawdown_pct = max_drawdown_pct   # halt trading if portfolio drops 15%
        self.position_size_pct = position_size_pct  # risk 10% of portfolio per trade
        self.stop_loss_pct = stop_loss_pct          # stop-loss at 5%

    def compute_position_size(self, portfolio_value: float, price: float) -> int:
        """Kelly-inspired fixed-fractional position sizing."""
        allocation = portfolio_value * self.position_size_pct
        return max(1, int(allocation / price))

    def check_drawdown(self, portfolio_value: float, peak_value: float) -> bool:
        """Returns True if trading should HALT due to max drawdown breach."""
        drawdown = (peak_value - portfolio_value) / peak_value
        return drawdown > self.max_drawdown_pct


# ─────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────

class Backtester:
    def __init__(self, initial_capital: float = 100_000, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost  # 0.1% per trade (realistic)
        self.risk_manager = RiskManager()

    def run(self, prices: pd.Series, signals: pd.Series, probas: pd.Series,
            threshold: float = 0.55) -> dict:
        """
        signals: 0/1 predictions aligned with prices index
        probas: model confidence [0,1]
        threshold: minimum confidence to act on signal
        """
        prices = prices.copy()
        signals = signals.copy()
        probas = probas.copy()

        portfolio_value = self.initial_capital
        peak_value = self.initial_capital
        cash = self.initial_capital
        shares = 0
        entry_price = None

        trades = []
        equity_curve = []
        halted = False

        for date, price in prices.items():
            if date not in signals.index:
                equity_curve.append({"date": date, "portfolio": cash + shares * price})
                continue

            signal = signals[date]
            proba = probas[date]
            current_portfolio = cash + shares * price

            # Update peak
            if current_portfolio > peak_value:
                peak_value = current_portfolio

            # Check max drawdown halt
            if self.risk_manager.check_drawdown(current_portfolio, peak_value):
                halted = True

            # Stop-loss: exit if in position and price dropped 5% from entry
            if shares > 0 and entry_price is not None:
                if (price - entry_price) / entry_price < -self.risk_manager.stop_loss_pct:
                    proceeds = shares * price * (1 - self.transaction_cost)
                    pnl = proceeds - (shares * entry_price)
                    trades.append({
                        "date": date, "type": "STOP_LOSS",
                        "price": price, "shares": shares, "pnl": pnl
                    })
                    cash += proceeds
                    shares = 0
                    entry_price = None

            if not halted:
                # BUY signal
                if signal == 1 and proba > threshold and shares == 0:
                    n = self.risk_manager.compute_position_size(cash, price)
                    cost = n * price * (1 + self.transaction_cost)
                    if cost <= cash:
                        cash -= cost
                        shares += n
                        entry_price = price
                        trades.append({
                            "date": date, "type": "BUY",
                            "price": price, "shares": n, "pnl": 0
                        })

                # SELL signal
                elif signal == 0 and shares > 0:
                    proceeds = shares * price * (1 - self.transaction_cost)
                    pnl = proceeds - (shares * entry_price)
                    trades.append({
                        "date": date, "type": "SELL",
                        "price": price, "shares": shares, "pnl": pnl
                    })
                    cash += proceeds
                    shares = 0
                    entry_price = None

            equity_curve.append({"date": date, "portfolio": cash + shares * price})

        # Final liquidation
        if shares > 0:
            final_price = prices.iloc[-1]
            cash += shares * final_price * (1 - self.transaction_cost)
            shares = 0

        equity_df = pd.DataFrame(equity_curve).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = self._compute_metrics(equity_df, trades_df)

        return {
            "equity_curve": equity_df,
            "trades": trades_df,
            "metrics": metrics,
            "halted": halted,
        }

    def _compute_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
        curve = equity_df["portfolio"]
        daily_returns = curve.pct_change().dropna()

        total_return = (curve.iloc[-1] / curve.iloc[0] - 1) * 100
        ann_return = ((curve.iloc[-1] / curve.iloc[0]) ** (252 / len(curve)) - 1) * 100
        sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
        rolling_max = curve.cummax()
        drawdown = (curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        volatility = daily_returns.std() * np.sqrt(252) * 100

        sortino = (daily_returns.mean() / (daily_returns[daily_returns < 0].std() + 1e-9)) * np.sqrt(252)

        win_rate, avg_win, avg_loss, profit_factor = 0, 0, 0, 0
        if not trades_df.empty:
            completed = trades_df[trades_df["type"].isin(["SELL", "STOP_LOSS"])]
            if len(completed):
                wins = completed[completed["pnl"] > 0]["pnl"]
                losses = completed[completed["pnl"] <= 0]["pnl"]
                win_rate = len(wins) / len(completed) * 100
                avg_win = wins.mean() if len(wins) else 0
                avg_loss = losses.mean() if len(losses) else 0
                profit_factor = wins.sum() / (abs(losses.sum()) + 1e-9)

        return {
            "Total Return (%)": round(total_return, 2),
            "Annualized Return (%)": round(ann_return, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Sortino Ratio": round(sortino, 3),
            "Max Drawdown (%)": round(max_drawdown, 2),
            "Volatility (%)": round(volatility, 2),
            "Win Rate (%)": round(win_rate, 2),
            "Avg Win ($)": round(avg_win, 2),
            "Avg Loss ($)": round(avg_loss, 2),
            "Profit Factor": round(profit_factor, 3),
            "Total Trades": len(trades_df[trades_df["type"] == "BUY"]) if not trades_df.empty else 0,
            "Final Portfolio ($)": round(curve.iloc[-1], 2),
        }


# ─────────────────────────────────────────────
# BUY-AND-HOLD BENCHMARK
# ─────────────────────────────────────────────

def benchmark_buy_hold(prices: pd.Series, initial_capital: float = 100_000) -> pd.DataFrame:
    shares = initial_capital / prices.iloc[0]
    equity = (prices * shares).to_frame("portfolio")
    return equity


# ─────────────────────────────────────────────
# RUN BACKTEST FROM PKL
# ─────────────────────────────────────────────

def run_backtest_from_pkl(pkl_path: str = "trading_model.pkl") -> dict:
    print(f"[BACKTEST] Loading {pkl_path}...")
    artifact = joblib.load(pkl_path)

    data = artifact["all_data"]
    test_data = artifact["test_data"]
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    proba = test_data["proba"]
    preds = test_data["preds"]

    prices = data.loc[X_test.index, "Close"]
    signals = pd.Series(preds, index=X_test.index)
    probas = pd.Series(proba, index=X_test.index)

    bt = Backtester(initial_capital=100_000)
    result = bt.run(prices, signals, probas, threshold=0.55)

    # Benchmark
    bh = benchmark_buy_hold(prices)
    result["benchmark"] = bh
    result["prices"] = prices

    print("\n[BACKTEST METRICS]")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    return result


if __name__ == "__main__":
    result = run_backtest_from_pkl("trading_model.pkl")
    print("\n[DONE] Backtest complete.")
