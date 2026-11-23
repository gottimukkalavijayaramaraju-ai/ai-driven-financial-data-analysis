"""AI/ML models: anomaly detection, next-day direction prediction, risk metrics."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

FEATURES = ["Return", "SMA_10", "SMA_30", "Volatility_10", "Momentum_10",
            "RSI_14", "MACD", "MACD_Signal", "Volume_Change"]


def detect_anomalies(df, contamination=0.02):
    """Flag unusual days (returns/volume/volatility) with Isolation Forest."""
    cols = ["Return", "Volume_Change", "Volatility_10"]
    X = StandardScaler().fit_transform(df[cols])
    model = IsolationForest(contamination=contamination, random_state=42)
    out = df.copy()
    out["Anomaly"] = model.fit_predict(X) == -1
    return out


def train_direction_model(df, test_size=0.2):
    """Predict whether tomorrow's close will be higher than today's (time-ordered split)."""
    d = df.copy()
    d["Target"] = (d["Close"].shift(-1) > d["Close"]).astype(int)
    d = d.iloc[:-1]
    split = int(len(d) * (1 - test_size))
    train, test = d.iloc[:split], d.iloc[split:]

    model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["Target"])
    pred = model.predict(test[FEATURES])

    metrics = {
        "accuracy": accuracy_score(test["Target"], pred),
        "report": classification_report(test["Target"], pred, zero_division=0),
        "importances": pd.Series(model.feature_importances_, index=FEATURES)
                         .sort_values(ascending=False),
    }
    latest = model.predict_proba(df[FEATURES].iloc[[-1]])[0]
    metrics["next_day_up_probability"] = float(latest[1])
    return model, metrics


def risk_metrics(df, risk_free=0.0):
    r = df["Return"].dropna()
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    drawdown = (cum / cum.cummax() - 1).min()
    return {
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": (ann_ret - risk_free) / ann_vol if ann_vol else np.nan,
        "Max Drawdown": drawdown,
        "VaR 95% (daily)": r.quantile(0.05),
    }
