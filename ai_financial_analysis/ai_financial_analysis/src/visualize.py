"""Charts saved to the outputs folder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_price_anomalies(df, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Close", lw=1.2)
    ax.plot(df.index, df["SMA_30"], label="SMA 30", lw=1)
    a = df[df["Anomaly"]]
    ax.scatter(a.index, a["Close"], color="red", zorder=5, label="Anomaly")
    ax.set_title("Price with Detected Anomalies")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_indicators(df, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df.index, df["RSI_14"]); axes[0].axhline(70, ls="--", c="r")
    axes[0].axhline(30, ls="--", c="g"); axes[0].set_title("RSI (14)")
    axes[1].plot(df.index, df["MACD"], label="MACD")
    axes[1].plot(df.index, df["MACD_Signal"], label="Signal")
    axes[1].legend(); axes[1].set_title("MACD")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_importances(imp, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    imp.sort_values().plot.barh(ax=ax)
    ax.set_title("Feature Importance (Random Forest)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
