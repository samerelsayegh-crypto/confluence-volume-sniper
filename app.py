import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import numpy as np
import yfinance as yf
from strategy import ConfluenceVolumeSniper

# =====================================================================
# --- Dashboard Configuration ---
# =====================================================================
st.set_page_config(layout="wide", page_title="Confluence & Volume Sniper", page_icon="🎯")

# Apply clean, modern light mode CSS with dynamic status colors
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Dynamic Status Banner Formatting */
    .status-green { padding: 15px; border-radius: 8px; background-color: #E8F5E9; color: #2E7D32; border-left: 5px solid #4CAF50; font-weight: bold; margin-bottom: 20px;}
    .status-orange { padding: 15px; border-radius: 8px; background-color: #FFF3E0; color: #E65100; border-left: 5px solid #FF9800; font-weight: bold; margin-bottom: 20px;}
    .status-red { padding: 15px; border-radius: 8px; background-color: #FFEBEE; color: #C62828; border-left: 5px solid #F44336; font-weight: bold; margin-bottom: 20px;}
    
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# --- Sidebar Setup & Navigation ---
# =====================================================================
st.sidebar.title("🎯 Sniper Engine")
page = st.sidebar.radio("Navigation", ["Home Dashboard", "Market Analytics", "Stock Testing"])

st.sidebar.markdown("---")
st.sidebar.subheader("Money Management")
account_balance = st.sidebar.number_input("Total Equity ($)", value=10000.0, step=1000.0)
risk_pct = st.sidebar.slider("Max Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)


# =====================================================================
# --- Page 1: Home Dashboard ---
# =====================================================================
if page == "Home Dashboard":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Single Asset Scan")
    symbol = st.sidebar.text_input("Ticker Symbol (e.g. AAPL, BTC-USD)", value="AAPL")
    interval = st.sidebar.selectbox("Operational Timeframe", ["15m", "30m", "1h", "1d"], index=2)
    period = st.sidebar.selectbox("Lookback Period", ["60d", "90d", "1y", "2y"], index=2)

    st.title("Confluence & Volume Sniper Dashboard")
    st.markdown("Automated Top-Down Analysis & Trade Execution Engine.")

    if st.sidebar.button("Execute Strategy Scan", use_container_width=True, type="primary"):
        with st.spinner(f"Running algorithmic scan for {symbol}..."):
            # Initialize Backend Object
            sniper = ConfluenceVolumeSniper(symbol=symbol, account_balance=account_balance, risk_pct=risk_pct/100.0)
            
            # Execute 5-Step Logic
            trend, result, df_daily, df_op = sniper.run(interval=interval, period=period)
            
            if df_op.empty:
                st.error(f"Failed to fetch data for {symbol}. Try checking the ticker name.")
                st.stop()
                
            # UI Top Row Metrics
            st.markdown("### Top-Down Trend Verification")
            c1, c2, c3 = st.columns(3)
            c1.metric("Macro Trend (Daily)", trend)
            c2.metric("Current Asset Price", f"${float(df_op['close'].iloc[-1]):.2f}")
            c3.metric("Analysis Timeframe", interval)
            
            st.markdown("---")
            
            # Output Trade Plan
            if result:
                st.markdown('<div class="status-green">🚨 VALID TRADE SETUP DETECTED AT CONFLUENCE ZONE 🚨</div>', unsafe_allow_html=True)
                
                st.subheader("Actionable Trading Plan")
                t1, t2, t3, t4, t5 = st.columns(5)
                t1.metric("Action Trigger", result["Action"])
                t2.metric("Aggressive Entry", f"${result['Aggressive Entry Price']}")
                t3.metric("Stop Loss", f"${result['Stop Loss']}")
                t4.metric("Pos Sizing (Units)", result['Position Size'])
                t5.metric("Target (1:2 RR)", f"${result['Target 1 (1:2 RR)']}")
                
                with st.expander("View Full Algorithmic Ledger Details", expanded=False):
                    st.json(result)
            else:
                if trend == "CHOPPY" or trend == "UNKNOWN":
                    st.markdown('<div class="status-red">⚠️ Market is actively CHOPPY. Sideways filter active. No trades recommended. Risk is too high.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-orange">ℹ️ No actionable Price Action trigger at active Confluence Zones. Waiting for pullback...</div>', unsafe_allow_html=True)
            
            # =====================================================================
            # --- Interactive Chart Plotting ---
            # =====================================================================
            st.markdown(f"### Live Chart & Dynamic Confluence Zones ({symbol.upper()})")
            
            # Trim plot to last 150 candles for readability
            df_plot = df_op.tail(150)
            
            # Create Subplots: Row 1 = Candlesticks + SMAs + Fibs, Row 2 = Volume
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            
            # Add Candlesticks
            fig.add_trace(go.Candlestick(
                x=df_plot.index, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'],
                name="PriceAction"
            ), row=1, col=1)
            
            # Add Dynamic S/R SMAs
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_8'], line=dict(color='#2196F3', width=1.5), name="8 SMA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_21'], line=dict(color='#FF9800', width=1.5), name="21 SMA"), row=1, col=1)
            
            # Volume VPA Bars
            colors = ['#4CAF50' if close >= open else '#F44336' for close, open in zip(df_plot['close'], df_plot['open'])]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], marker_color=colors, name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Vol_SMA'], line=dict(color='#FFEB3B', width=2), name="20 Vol SMA"), row=2, col=1)
            
            fig.update_layout(
                height=700, 
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#1E1E1E")
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Configure parameters on the left and click 'Execute Strategy Scan' to begin your analysis.")

# =====================================================================
# --- Page 2: Market Analytics ---
# =====================================================================
elif page == "Market Analytics":
    st.title("📈 Market Analytics Scanner")
    st.markdown("Scan a predefined watchlist of assets to instantly discover actionable trade setups based on your Confluence & Volume Sniper strategy.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Scanner Settings")
    scan_interval = st.sidebar.selectbox("Scan Timeframe", ["15m", "30m", "1h", "1d"], index=3) # Default 1d
    scan_period = st.sidebar.selectbox("Lookback Period", ["60d", "90d", "1y", "2y"], index=3) # Default 2y
    
    # Predefined High-Volume Watchlist (Mega-caps, highly liquid tech, high beta)
    default_watchlist = (
        "AAPL, NVDA, TSLA, AMD, AMZN, MSFT, META, GOOGL, PLTR, SOFI, "
        "MARA, COIN, INTC, MU, BAC, C, F, T, VZ, PFE, CSCO, DIS, NFLX, "
        "UBER, BA, WMT, XOM, CVX, JPM, V, MA, PYPL, SQ, CRM, ADBE, "
        "QCOM, TXN, AVGO, SBUX, NKE, KO, PEP, MCD, HD, LOW, TGT"
    )
    watchlist_input = st.sidebar.text_area("Watchlist (Comma Separated)", value=default_watchlist, height=150)
    
    if st.button("Run System-Wide Scan", use_container_width=True, type="primary"):
        tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
        
        if not tickers:
            st.warning("Please enter at least one ticker.")
            st.stop()
            
        st.markdown("### 🔍 Scan Results")
        
        # Track setups found
        setups_found = []
        
        progress_text = "Scanning Watchlist. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        for i, ticker in enumerate(tickers):
            # Update progress bar
            percent_complete = int(((i) / len(tickers)) * 100)
            my_bar.progress(percent_complete, text=f"Scanning {ticker}... ({i+1}/{len(tickers)})")
            
            # Run engine
            sniper = ConfluenceVolumeSniper(symbol=ticker, account_balance=account_balance, risk_pct=risk_pct/100.0)
            trend, result, df_daily, df_op = sniper.run(interval=scan_interval, period=scan_period)
            
            if result:
                result['Ticker'] = ticker
                setups_found.append(result)
                
            time.sleep(0.1) # Prevents YFinance rate limits
            
        my_bar.progress(100, text="Scan Complete!")
        
        if setups_found:
            st.markdown(f'<div class="status-green">🎯 Found {len(setups_found)} Actionable Setup(s)!</div>', unsafe_allow_html=True)
            
            for setup in setups_found:
                with st.container():
                    st.markdown(f"#### {setup['Ticker']} - {setup['Action']} Signal")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Trend Alignment", setup.get("Trend", "UNKNOWN"))
                    col2.metric("Entry Price", f"${setup['Aggressive Entry Price']}")
                    col3.metric("Stop Loss", f"${setup['Stop Loss']}")
                    col4.metric("Pos Sizing", setup['Position Size'])
                    col5.metric("Target (1:2)", f"${setup['Target 1 (1:2 RR)']}")
                    st.markdown("---")
        else:
            st.markdown('<div class="status-orange">ℹ️ No actionable setups found across the watchlist on this timeframe. The broader market may be choppy or assets are waiting for algorithmic pullbacks.</div>', unsafe_allow_html=True)

# =====================================================================
# --- Page 3: Stock Testing ---
# =====================================================================
elif page == "Stock Testing":
    st.title("🧪 Stock Testing: QQQ 5-Minute Monitor")
    st.markdown("Calculates the 9 EMA, 20 EMA, and 50 WMA to determine current market status.")
    
    if st.button("Run Monitor Update", type="primary"):
        with st.spinner("Fetching QQQ 5m data..."):
            try:
                # yfinance limits 5m data to 60 days
                df = yf.download("QQQ", interval="5m", period="60d", progress=False)
                
                if df.empty:
                    st.error("Failed to fetch QQQ data. Current market might be closed or hit rate limits.")
                else:
                    # Format columns (yfinance sometimes returns MultiIndex columns if single ticker downloaded)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0].lower() for col in df.columns]
                    else:
                        df.columns = [col.lower() for col in df.columns]
                        
                    # Calculate 9 EMA
                    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
                    
                    # Calculate 20 EMA
                    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
                    
                    # Calculate 50 WMA
                    weights = np.arange(1, 51)
                    df['WMA_50'] = df['close'].rolling(50).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
                    
                    # Get latest values for visual dashboard
                    latest = df.dropna().iloc[-1]
                    ema9 = latest['EMA_9']
                    ema20 = latest['EMA_20']
                    wma50 = latest['WMA_50']
                    close_price = latest['close']
                    
                    # Dashboard Logic (table.new equivalent)
                    if ema9 > ema20 and ema20 > wma50:
                        status = "UPTREND"
                        css_class = "status-green"
                        message = "🟢 9 EMA > 20 EMA > 50 WMA. Perfect Bullish Stack Active."
                    elif ema9 < ema20 and ema20 < wma50:
                        status = "DOWNTREND"
                        css_class = "status-red"
                        message = "🔴 9 EMA < 20 EMA < 50 WMA. Perfect Bearish Stack Active."
                    else:
                        status = "CHOP / TRANSITION"
                        css_class = "status-orange"
                        message = "🟡 Moving averages intertwined. Market in transition or choppy sideways."
                    
                    st.markdown("### Visual Status Dashboard")
                    st.markdown(f'<div class="{css_class}" style="text-align: center; font-size: 20px;">{message}</div>', unsafe_allow_html=True)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("QQQ Current Price", f"${close_price:.2f}")
                    c2.metric("9 EMA", f"{ema9:.2f}")
                    c3.metric("20 EMA", f"{ema20:.2f}")
                    c4.metric("50 WMA", f"{wma50:.2f}")
                    
                    st.markdown("---")
                    st.subheader("Raw Technical Data (Last 5 periods)")
                    st.dataframe(df[['close', 'EMA_9', 'EMA_20', 'WMA_50']].dropna().tail(5), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error executing QQQ monitor: {e}")
