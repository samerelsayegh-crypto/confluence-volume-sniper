# Confluence & Volume Sniper

A professional quantitative trading dashboard built with Streamlit and Python. It implements a 5-step strategic framework:
1. Top-Down Trend Analysis (Daily Timeframe)
2. Dynamic Confluence Zones (SMAs & Fibonacci)
3. Price Action Signal Detection
4. Volume Price Analysis (VPA)
5. Algorithmic Trade Plan Generation (Position Sizing, SL, TP)

## Features
- **Real-Time Data**: Fetches live market data using `yfinance`.
- **Dynamic Charting**: Interactive Plotly candlestick charts with volume profiles and moving averages.
- **Multi-Asset Analytics**: Includes a Watchlist Scanner to quickly detect setups across multiple assets.
- **Risk Management**: Strictly adheres to 1-2% risk per trade with dynamic position sizing.

## Local Deployment
```bash
pip install -r requirements.txt
streamlit run app.py
```
