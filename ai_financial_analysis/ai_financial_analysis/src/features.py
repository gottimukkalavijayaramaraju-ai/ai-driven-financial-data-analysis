"""Technical indicator feature engineering."""
import numpy as np
import pandas as pd


def add_features(df):
    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["SMA_10"] = d["Close"].rolling(10).mean()
    d["SMA_30"] = d["Close"].rolling(30).mean()
    d["Volatility_10"] = d["Return"].rolling(10).std()
    d["Momentum_10"] = d["Close"] / d["Close"].shift(10) - 1

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    d["Volume_Change"] = d["Volume"].pct_change()
    return d.dropna()
