# AI-Driven Financial Data Analysis

A simple base project that applies machine learning to financial time-series data.

## Features
- Data loading: CSV, Yahoo Finance (optional), or built-in synthetic generator (offline)
- Technical indicators: SMA, RSI, MACD, volatility, momentum
- Anomaly detection (Isolation Forest) for unusual price/volume days
- Next-day direction prediction (Random Forest, time-ordered train/test split)
- Risk metrics: annual return/volatility, Sharpe ratio, max drawdown, VaR
- Charts + CSV results saved to `outputs/`

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py                        # synthetic data
python main.py --csv data/sample.csv  # your CSV (Date,Open,High,Low,Close,Volume)
python main.py --ticker AAPL          # live data
```

## Structure
```
main.py            entry point
src/data_loader.py data sources
src/features.py    indicators
src/models.py      ML models + risk metrics
src/visualize.py   plots
data/              input data
outputs/           charts & results
```

## Ideas for extension
LSTM forecasting, news sentiment (FinBERT), portfolio optimization, Streamlit dashboard.

> Educational use only. Not financial advice.
