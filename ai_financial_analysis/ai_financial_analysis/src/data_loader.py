"""Load financial data: from CSV, Yahoo Finance (optional), or synthetic generator."""
import numpy as np
import pandas as pd


def generate_synthetic(days=750, seed=42, start_price=100.0):
    """Create realistic-looking OHLCV data using geometric Brownian motion + a few shocks."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.012, days)
    shock_idx = rng.choice(days, 6, replace=False)       # inject anomalies
    returns[shock_idx] += rng.choice([-0.08, 0.08], 6)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.002, days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, days))
    volume = rng.integers(1_000_000, 5_000_000, days).astype(float)
    volume[shock_idx] *= 3
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)


def load_csv(path):
    """CSV needs columns: Date, Open, High, Low, Close, Volume."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def load_yahoo(ticker="AAPL", period="3y"):
    import yfinance as yf  # needs internet
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
