"""AI-Driven Financial Data Analysis - entry point.

Examples:
    python main.py                          # synthetic data (works offline)
    python main.py --csv data/sample.csv    # your own CSV
    python main.py --ticker AAPL            # live data via Yahoo Finance
"""
import argparse
import os

from src.data_loader import generate_synthetic, load_csv, load_yahoo
from src.features import add_features
from src.models import detect_anomalies, train_direction_model, risk_metrics
from src.visualize import plot_price_anomalies, plot_indicators, plot_importances


def main():
    p = argparse.ArgumentParser(description="AI-Driven Financial Data Analysis")
    p.add_argument("--csv", help="Path to OHLCV CSV file")
    p.add_argument("--ticker", help="Ticker symbol (requires yfinance + internet)")
    p.add_argument("--out", default="outputs", help="Output folder")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.csv:
        raw = load_csv(args.csv)
    elif args.ticker:
        raw = load_yahoo(args.ticker)
    else:
        raw = generate_synthetic()
        raw.to_csv("data/sample.csv", index_label="Date")
        print("Using synthetic data (saved to data/sample.csv)")

    df = add_features(raw)
    print(f"Loaded {len(df)} rows: {df.index[0].date()} -> {df.index[-1].date()}\n")

    print("== Risk Metrics ==")
    for k, v in risk_metrics(df).items():
        print(f"{k:20s}: {v:8.3f}")

    df = detect_anomalies(df)
    print(f"\n== Anomalies detected: {int(df['Anomaly'].sum())} days ==")
    print(df[df["Anomaly"]][["Close", "Return"]].tail(10).round(4))

    _, m = train_direction_model(df)
    print(f"\n== Next-day Direction Model ==\nTest accuracy: {m['accuracy']:.2%}")
    print(m["report"])
    print(f"Probability price rises tomorrow: {m['next_day_up_probability']:.1%}")
    print("(Educational only - not financial advice.)")

    plot_price_anomalies(df, f"{args.out}/price_anomalies.png")
    plot_indicators(df, f"{args.out}/indicators.png")
    plot_importances(m["importances"], f"{args.out}/feature_importance.png")
    df.to_csv(f"{args.out}/analysis_results.csv")
    print(f"\nCharts and results saved in '{args.out}/'")


if __name__ == "__main__":
    main()
