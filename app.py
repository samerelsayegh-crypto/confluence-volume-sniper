import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
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
page = st.sidebar.radio("Navigation", ["Home Dashboard", "Market Analytics", "Signal 1 - MA", "Signal 2 - 15 Min ORB", "Signal 3 - VWAP", "Signal 4 - PD High / Low Distances", "Signal 5 - PM High / Low Prices"])

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
# --- Page 3: Signal 1 - MA ---
# =====================================================================
elif page == "Signal 1 - MA":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Testing Parameters")
    test_symbol = st.sidebar.text_input("Test Ticker", value="QQQ").upper()
    
    st.title(f"🧪 Signal 1 - MA: {test_symbol} Multi-Timeframe Monitor")
    st.markdown("Real-time automated evaluation across 5m, 15m, 1h, and 1d timeframes.")
    
    if st.button("Run Multi-Timeframe Scan", type="primary"):
        with st.spinner(f"Fetching and analyzing multi-timeframe data for {test_symbol}..."):
            
            timeframes = ["5m", "15m", "1h", "1d"]
            results = {}
            
            for tf in timeframes:
                try:
                    alpaca_key = st.secrets["alpaca"]["API_KEY"]
                    alpaca_secret = st.secrets["alpaca"]["API_SECRET"]
                    client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
                    
                    # Map timeframe and lookback
                    if tf == "5m":
                        alpaca_tf = TimeFrame(5, TimeFrameUnit.Minute)
                        lookback_days = 60
                    elif tf == "15m":
                        alpaca_tf = TimeFrame(15, TimeFrameUnit.Minute)
                        lookback_days = 60
                    elif tf == "1h":
                        alpaca_tf = TimeFrame(1, TimeFrameUnit.Hour)
                        lookback_days = 730
                    else: # 1d
                        alpaca_tf = TimeFrame(1, TimeFrameUnit.Day)
                        lookback_days = 730
                        
                    now = datetime.now(timezone.utc)
                    start_dt = now - timedelta(days=lookback_days)
                    
                    req = StockBarsRequest(
                        symbol_or_symbols=test_symbol,
                        timeframe=alpaca_tf,
                        start=start_dt,
                        end=now - timedelta(minutes=16), # Free tier is delayed by 15 mins
                        feed=DataFeed.IEX
                    )
                    
                    bars = client.get_stock_bars(req)
                    if not bars or bars.df.empty:
                        results[tf] = {"error": True}
                        continue
                        
                    df = bars.df.droplevel(0)
                    
                    # Filter for Regular Market Hours (9:30 AM to 4:00 PM EST) if intraday
                    if tf != "1d":
                        df.index = df.index.tz_convert('America/New_York')
                        df = df.between_time('09:30', '16:00')
                        
                    # Calculate MAs
                    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
                    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
                    
                    weights_50 = np.arange(1, 51)
                    df['WMA_50'] = df['close'].rolling(50).apply(lambda prices: np.dot(prices, weights_50)/weights_50.sum(), raw=True)
                    
                    weights_100 = np.arange(1, 101)
                    df['WMA_100'] = df['close'].rolling(100).apply(lambda prices: np.dot(prices, weights_100)/weights_100.sum(), raw=True)
                    
                    weights_200 = np.arange(1, 201)
                    df['WMA_200'] = df['close'].rolling(200).apply(lambda prices: np.dot(prices, weights_200)/weights_200.sum(), raw=True)
                    
                    # Ensure we have enough data points to compute the 200 WMA
                    df_clean = df.dropna()
                    if len(df_clean) == 0:
                        results[tf] = {"error": True}
                        continue
                        
                    latest = df_clean.iloc[-1]
                    ema9 = latest['EMA_9']
                    ema20 = latest['EMA_20']
                    wma50 = latest['WMA_50']
                    wma100 = latest['WMA_100']
                    wma200 = latest['WMA_200']
                    close_price = latest['close']
                    
                    res_tf = {"error": False, "price": close_price}
                    
                    # Logic 1: Short-Term Stack
                    if ema9 > ema20 and ema20 > wma50:
                        res_tf["st_stack"] = {"status": "UPTREND", "css": "status-green", "icon": "🟢"}
                    elif ema9 < ema20 and ema20 < wma50:
                        res_tf["st_stack"] = {"status": "DOWNTREND", "css": "status-red", "icon": "🔴"}
                    else:
                        res_tf["st_stack"] = {"status": "CHOP", "css": "status-orange", "icon": "🟡"}
                        
                    # Logic 2: Price vs ST MAs
                    if close_price > ema9 and close_price > ema20 and close_price > wma50:
                        res_tf["price_st"] = {"status": "UPTREND", "css": "status-green", "icon": "🟢"}
                    elif close_price < ema9 and close_price < ema20 and close_price < wma50:
                        res_tf["price_st"] = {"status": "DOWNTREND", "css": "status-red", "icon": "🔴"}
                    else:
                        res_tf["price_st"] = {"status": "MIXED", "css": "status-orange", "icon": "🟡"}
                        
                    # Logic 3: Long-Term Trend
                    if wma100 > wma200:
                        res_tf["lt_trend"] = {"status": "UPTREND", "css": "status-green", "icon": "🟢"}
                    elif wma100 < wma200:
                        res_tf["lt_trend"] = {"status": "DOWNTREND", "css": "status-red", "icon": "🔴"}
                    else:
                        res_tf["lt_trend"] = {"status": "NEUTRAL", "css": "status-orange", "icon": "🟡"}
                        
                    # Logic 4: Price vs LT Trend
                    if close_price > wma100 and close_price > wma200:
                        res_tf["price_lt"] = {"status": "UPTREND", "css": "status-green", "icon": "🟢"}
                    elif close_price < wma100 and close_price < wma200:
                        res_tf["price_lt"] = {"status": "DOWNTREND", "css": "status-red", "icon": "🔴"}
                    else:
                        res_tf["price_lt"] = {"status": "MIXED", "css": "status-orange", "icon": "🟡"}
                        
                    # Save the dataframe for the chart and raw data display (last 100 periods to keep it fast)
                    res_tf["df"] = df_clean.tail(100)
                        
                    results[tf] = res_tf
                    
                except Exception as e:
                    st.error(f"Error processing {tf}: {e}")
                    results[tf] = {"error": True}
                    
            st.markdown("### Unified Multi-Timeframe Grid")
            
            # Helper function to render a cell
            def render_cell(res_data, key):
                if res_data.get("error"):
                    return '<div style="padding:15px; border-radius:8px; background-color:#f5f5f5; color:#999; text-align:center; font-weight:bold; height: 100%;">N/A</div>'
                
                cell = res_data[key]
                # Reusing the existing CSS classes for styling backgrounds
                return f'<div class="{cell["css"]}" style="text-align:center; padding:15px; margin-bottom:0px; height: 100%; display: flex; align-items: center; justify-content: center;">{cell["icon"]} {cell["status"]}</div>'
                
            # Create Grid using st.columns
            # Layout: [Metric Name (2)] [5m (1)] [15m (1)] [1h (1)] [1d (1)]
            header_cols = st.columns([2, 1, 1, 1, 1])
            header_cols[0].markdown("**Indicator / Timeframe**")
            header_cols[1].markdown('<div style="text-align:center;"><b>5m</b></div>', unsafe_allow_html=True)
            header_cols[2].markdown('<div style="text-align:center;"><b>15m</b></div>', unsafe_allow_html=True)
            header_cols[3].markdown('<div style="text-align:center;"><b>1h</b></div>', unsafe_allow_html=True)
            header_cols[4].markdown('<div style="text-align:center;"><b>1d</b></div>', unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            
            # Row 1
            r1 = st.columns([2, 1, 1, 1, 1], vertical_alignment="center")
            r1[0].markdown("**1. Short-Term MA Stack**<br/><small>(9 EMA vs 20 EMA vs 50 WMA)</small>", unsafe_allow_html=True)
            for i, tf in enumerate(timeframes):
                r1[i+1].markdown(render_cell(results[tf], "st_stack"), unsafe_allow_html=True)
                
            st.markdown("<br/>", unsafe_allow_html=True)
            
            # Row 2
            r2 = st.columns([2, 1, 1, 1, 1], vertical_alignment="center")
            r2[0].markdown("**2. Price vs Averages**<br/><small>(Current Price vs 9, 20, 50)</small>", unsafe_allow_html=True)
            for i, tf in enumerate(timeframes):
                r2[i+1].markdown(render_cell(results[tf], "price_st"), unsafe_allow_html=True)
                
            st.markdown("<br/>", unsafe_allow_html=True)
                
            # Row 3
            r3 = st.columns([2, 1, 1, 1, 1], vertical_alignment="center")
            r3[0].markdown("**3. Long-Term Trend**<br/><small>(100 WMA vs 200 WMA)</small>", unsafe_allow_html=True)
            for i, tf in enumerate(timeframes):
                r3[i+1].markdown(render_cell(results[tf], "lt_trend"), unsafe_allow_html=True)
                
            st.markdown("<br/>", unsafe_allow_html=True)
                
            # Row 4
            r4 = st.columns([2, 1, 1, 1, 1], vertical_alignment="center")
            r4[0].markdown("**4. Price vs Long-Term Trend**<br/><small>(Current Price vs 100, 200)</small>", unsafe_allow_html=True)
            for i, tf in enumerate(timeframes):
                r4[i+1].markdown(render_cell(results[tf], "price_lt"), unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Print latest prices
            st.markdown("### Current Asset Reference")
            p_cols = st.columns(4)
            for i, tf in enumerate(timeframes):
                if not results[tf].get("error"):
                    p_cols[i].metric(f"{test_symbol} Close Price ({tf})", f"${results[tf]['price']:.2f}")

            st.markdown("---")
            st.markdown(f"### 📈 Technical Details & Charts")
            
            # Create tabs for each timeframe to show charts and raw data
            tabs = st.tabs([f"{tf} Data" for tf in timeframes])
            
            for i, tf in enumerate(timeframes):
                with tabs[i]:
                    if results[tf].get("error"):
                        st.warning(f"No valid data could be calculated for the {tf} timeframe.")
                    else:
                        hist_df = results[tf]["df"]
                        
                        # Render Chart
                        fig = go.Figure()
                        
                        # Add Price
                        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['close'], mode='lines', name='Close Price', line=dict(color='black', width=2)))
                        # Add MAs
                        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['EMA_9'], mode='lines', name='9 EMA', line=dict(color='blue', width=1)))
                        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['EMA_20'], mode='lines', name='20 EMA', line=dict(color='orange', width=1)))
                        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['WMA_50'], mode='lines', name='50 WMA', line=dict(color='purple', width=2)))
                        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['WMA_100'], mode='lines', name='100 WMA', line=dict(color='red', width=2)))
                        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['WMA_200'], mode='lines', name='200 WMA', line=dict(color='green', width=3, dash='dot')))
                        
                        fig.update_layout(
                            title=f"{test_symbol} - {tf} Chart with Moving Averages",
                            xaxis_title="Time",
                            yaxis_title="Price",
                            template="plotly_white",
                            height=500,
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Render Data
                        st.markdown(f"**Latest Raw Data ({tf})**")
                        # Show only the relevant columns and the last 15 rows for neatness
                        display_df = hist_df[['close', 'EMA_9', 'EMA_20', 'WMA_50', 'WMA_100', 'WMA_200']].tail(15)
                        st.dataframe(display_df.style.format("{:.2f}"), use_container_width=True)
                        
                        # Add Data Source Attribution
                        st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# --- Page 4: Signal 2 - 15 Min ORB ---
# =====================================================================
elif page == "Signal 2 - 15 Min ORB":
    st.sidebar.markdown("---")
    st.sidebar.subheader("ORB Parameters")
    test_symbol = st.sidebar.text_input("Ticker", value="QQQ").upper()
    
    st.title(f"🚀 Signal 2: {test_symbol} 15-Minute ORB Strategy")
    st.markdown("Automated detection of the Opening Range Breakout (09:30 - 09:45 EST). Monitors 5-minute candles for breakout signals.")
    
    if st.button("Detect ORB Signals", type="primary"):
        with st.spinner(f"Analyzing {test_symbol} 5m footprint..."):
            try:
                alpaca_key = st.secrets["alpaca"]["API_KEY"]
                alpaca_secret = st.secrets["alpaca"]["API_SECRET"]
                client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
                
                # Fetch 5-minute data for the last 5 days to ensure we have recent sessions
                now = datetime.now(timezone.utc)
                start_dt = now - timedelta(days=5)
                
                req = StockBarsRequest(
                    symbol_or_symbols=test_symbol,
                    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                    start=start_dt,
                    end=now - timedelta(minutes=16),
                    feed=DataFeed.IEX
                )
                
                bars = client.get_stock_bars(req)
                if not bars or bars.df.empty:
                    st.error("No data found or market is closed.")
                    st.stop()
                    
                df = bars.df.droplevel(0)
                df.index = df.index.tz_convert('America/New_York')
                
                # Group by date to find ORB for each day
                df['Date'] = df.index.date
                df['Time'] = df.index.time
                
                # Separate into days
                unique_days = df['Date'].unique()
                
                # Get the latest day with data
                latest_day = unique_days[-1]
                df_latest_day = df[df['Date'] == latest_day].copy()
                
                # Filter strictly to Regular Market Hours just so charts/signals are clean
                df_latest_day = df_latest_day.between_time('09:30', '16:00')
                
                # Filter 09:30 to 09:45 for the Opening Range
                # 5-min bars: 09:30, 09:35, 09:40. The 09:40 bar closes at 09:45.
                orb_bars = df_latest_day.between_time('09:30', '09:40')
                
                if orb_bars.empty or len(orb_bars) < 3:
                    st.warning(f"Not enough data to calculate the 15-minute ORB for {latest_day}. Market may just have opened.")
                    st.stop()
                    
                orb_high = orb_bars['high'].max()
                orb_low = orb_bars['low'].min()
                
                # Data *after* the opening range (09:45 onwards)
                post_orb_bars = df_latest_day.between_time('09:45', '16:00').copy()
                
                st.markdown(f"### {test_symbol} ORB Levels ({latest_day})")
                c1, c2 = st.columns(2)
                c1.metric("15m ORB High", f"${orb_high:.2f}")
                c2.metric("15m ORB Low", f"${orb_low:.2f}")
                
                st.markdown("---")
                
                # Detect Signals
                signals = []
                for timestamp, row in post_orb_bars.iterrows():
                    if row['close'] > orb_high:
                        signals.append((timestamp, "BULLISH BREAKOUT", row['close'], 'status-green', '🟢'))
                        # Only take the first signal of the day
                        break
                    elif row['close'] < orb_low:
                        signals.append((timestamp, "BEARISH BREAKDOWN", row['close'], 'status-red', '🔴'))
                        break
                        
                if signals:
                    sig_time, sig_type, sig_price, css_class, icon = signals[0]
                    st.markdown(f'<div class="{css_class}">{icon} {sig_type} SIGNAL DETECTED</div>', unsafe_allow_html=True)
                    st.markdown(f"**Trigger Time:** {sig_time.strftime('%H:%M EST')}")
                    st.markdown(f"**Trigger Price (5m Close):** ${sig_price:.2f}")
                else:
                    st.markdown('<div class="status-orange">🟡 NO SIGNAL: Price is trading within the Opening Range or no 5m candle has closed outside it yet.</div>', unsafe_allow_html=True)
                    
                # Plotting
                st.markdown("---")
                st.markdown("### Interactive ORB Chart")
                
                fig = go.Figure()
                
                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=df_latest_day.index, open=df_latest_day['open'], high=df_latest_day['high'], 
                    low=df_latest_day['low'], close=df_latest_day['close'], name="Price"
                ))
                
                # ORB Lines
                fig.add_hline(y=orb_high, line_dash="dash", line_color="green", annotation_text="ORB High")
                fig.add_hline(y=orb_low, line_dash="dash", line_color="red", annotation_text="ORB Low")
                
                # Highlight ORB Zone
                fig.add_vrect(
                    x0=orb_bars.index[0], x1=orb_bars.index[-1] + timedelta(minutes=5),
                    fillcolor="LightSalmon", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text="ORB Period", annotation_position="top left"
                )
                
                # Highlight Signal if any
                if signals:
                    fig.add_annotation(
                        x=sig_time, y=sig_price,
                        text=sig_type,
                        showarrow=True, arrowhead=1, ax=0, ay=-40 if sig_type == "BULLISH BREAKOUT" else 40,
                        bgcolor="green" if sig_type == "BULLISH BREAKOUT" else "red",
                        font=dict(color="white")
                    )

                fig.update_layout(
                    height=600, 
                    xaxis_rangeslider_visible=False,
                    template="plotly_white",
                    xaxis_title="Time (EST)",
                    yaxis_title="Price",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                    
                st.plotly_chart(fig, use_container_width=True)
                
                # Raw data
                with st.expander("View Raw Intraday Data"):
                    display_df = df_latest_day[['open', 'high', 'low', 'close', 'volume']]
                    st.dataframe(display_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing ORB: {e}")

# =====================================================================
# --- Page 5: Signal 3 - VWAP ---
# =====================================================================
elif page == "Signal 3 - VWAP":
    st.sidebar.markdown("---")
    st.sidebar.subheader("VWAP Parameters")
    test_symbol = st.sidebar.text_input("Ticker", value="QQQ").upper()
    
    st.title(f"⚖️ Signal 3: {test_symbol} Daily Anchored VWAP")
    st.markdown("Automated detection of the current price relative to the Daily Volume Weighted Average Price (VWAP).")
    
    if st.button("Calculate VWAP", type="primary"):
        with st.spinner(f"Analyzing {test_symbol} VWAP..."):
            try:
                alpaca_key = st.secrets["alpaca"]["API_KEY"]
                alpaca_secret = st.secrets["alpaca"]["API_SECRET"]
                client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
                
                # Fetch 5-minute data for the current/latest day
                now = datetime.now(timezone.utc)
                start_dt = now - timedelta(days=5) # Get a few days to ensure we have the latest session
                
                req = StockBarsRequest(
                    symbol_or_symbols=test_symbol,
                    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                    start=start_dt,
                    end=now - timedelta(minutes=16),
                    feed=DataFeed.IEX
                )
                
                bars = client.get_stock_bars(req)
                if not bars or bars.df.empty:
                    st.error("No data found or market is closed.")
                    st.stop()
                    
                df = bars.df.droplevel(0)
                df.index = df.index.tz_convert('America/New_York')
                
                # Filter strictly to Regular Market Hours
                df = df.between_time('09:30', '16:00')
                
                # Group by date
                df['Date'] = df.index.date
                
                # Get the latest day with data
                unique_days = df['Date'].unique()
                latest_day = unique_days[-1]
                df_latest_day = df[df['Date'] == latest_day].copy()
                
                if df_latest_day.empty:
                    st.warning("No data found for the regular trading session today.")
                    st.stop()
                    
                # Calculate Daily Anchored VWAP
                # Typical Price = (High + Low + Close) / 3
                df_latest_day['Typical_Price'] = (df_latest_day['high'] + df_latest_day['low'] + df_latest_day['close']) / 3
                df_latest_day['VP'] = df_latest_day['Typical_Price'] * df_latest_day['volume']
                
                df_latest_day['Cumulative_VP'] = df_latest_day['VP'].cumsum()
                df_latest_day['Cumulative_Volume'] = df_latest_day['volume'].cumsum()
                
                df_latest_day['VWAP'] = df_latest_day['Cumulative_VP'] / df_latest_day['Cumulative_Volume']
                
                # Current Price & VWAP
                current_price = df_latest_day['close'].iloc[-1]
                current_vwap = df_latest_day['VWAP'].iloc[-1]
                
                st.markdown(f"### {test_symbol} VWAP Status ({latest_day})")
                c1, c2, c3 = st.columns(3)
                c1.metric("Live Price (Last Close)", f"${current_price:.2f}")
                c2.metric("Daily VWAP", f"${current_vwap:.2f}")
                
                # Determine relationship
                diff = current_price - current_vwap
                pct_diff = (diff / current_vwap) * 100
                c3.metric("Distance from VWAP", f"${diff:.2f}", f"{pct_diff:.2f}%")
                
                st.markdown("---")
                
                if current_price > current_vwap:
                    st.markdown('<div class="status-green">🟢 BULLISH: Price is holding ABOVE the Daily VWAP. Buyers are in control.</div>', unsafe_allow_html=True)
                elif current_price < current_vwap:
                    st.markdown('<div class="status-red">🔴 BEARISH: Price is holding BELOW the Daily VWAP. Sellers are in control.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-orange">🟡 NEUTRAL: Price is exactly at VWAP.</div>', unsafe_allow_html=True)
                    
                # Plotting
                st.markdown("---")
                st.markdown("### Interactive VWAP Chart")
                
                fig = go.Figure()
                
                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=df_latest_day.index, open=df_latest_day['open'], high=df_latest_day['high'], 
                    low=df_latest_day['low'], close=df_latest_day['close'], name="Price"
                ))
                
                # VWAP Line
                fig.add_trace(go.Scatter(
                    x=df_latest_day.index, y=df_latest_day['VWAP'], 
                    mode='lines', name='VWAP', line=dict(color='purple', width=3)
                ))
                
                fig.update_layout(
                    height=600, 
                    xaxis_rangeslider_visible=False,
                    template="plotly_white",
                    xaxis_title="Time (EST)",
                    yaxis_title="Price",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                    
                st.plotly_chart(fig, use_container_width=True)
                
                # Raw data
                with st.expander("View Raw Intraday Data"):
                    # Format to 2 decimal places before display
                    display_df = df_latest_day[['open', 'high', 'low', 'close', 'volume', 'VWAP']].copy()
                    display_df['VWAP'] = display_df['VWAP'].round(2)
                    st.dataframe(display_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing VWAP: {e}")

# =====================================================================
# --- Page 6: Signal 4 - PD High / Low Distances ---
# =====================================================================
elif page == "Signal 4 - PD High / Low Distances":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Watchlist Settings")
    default_wl = "QQQ, SPY, TSLA, NVDA, AAPL, MSFT, META, GOOGL, AMZN, PLTR, COIN, MSTR"
    watchlist_input = st.sidebar.text_area("Tickers (Comma Separated)", value=default_wl, height=100)
    
    st.title("📊 Signal 4: Daily High & Low Divergence Tracker")
    st.markdown("Monitor real-time percentage distances from the Daily High, Daily Low, and Yesterday's Close across an entire watchlist.")
    
    if st.button("Run Watchlist Scan", type="primary"):
        tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
        
        if not tickers:
            st.warning("Please enter at least one ticker.")
            st.stop()
            
        with st.spinner("Fetching market data..."):
            try:
                alpaca_key = st.secrets["alpaca"]["API_KEY"]
                alpaca_secret = st.secrets["alpaca"]["API_SECRET"]
                client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
                
                # We fetch 1-Day timeframe data for the last 5 days to get yesterday's close and today's intraday high/low
                now = datetime.now(timezone.utc)
                start_dt = now - timedelta(days=5)
                
                req = StockBarsRequest(
                    symbol_or_symbols=tickers,
                    timeframe=TimeFrame(1, TimeFrameUnit.Day),
                    start=start_dt,
                    end=now,
                    feed=DataFeed.IEX
                )
                
                bars = client.get_stock_bars(req)
                if not bars or bars.df.empty:
                    st.error("No data found.")
                    st.stop()
                    
                df_all = bars.df.reset_index()
                
                # Ensure timezone awareness for filtering
                if df_all['timestamp'].dt.tz is None:
                    df_all['timestamp'] = df_all['timestamp'].dt.tz_localize('UTC')
                
                # Get the latest two unique dates
                df_all['Date'] = df_all['timestamp'].dt.date
                unique_dates = sorted(list(set(df_all['Date'])))
                
                if len(unique_dates) < 1:
                    st.warning("Not enough data to compute prior day comparisons.")
                    st.stop()
                    
                latest_date = unique_dates[-1]
                prev_date = unique_dates[-2] if len(unique_dates) > 1 else None
                
                st.markdown(f"**Data representing trading session for:** {latest_date}")
                
                results = []
                
                # Create a progress bar
                progress_text = "Processing metrics..."
                my_bar = st.progress(0, text=progress_text)
                
                for i, ticker in enumerate(tickers):
                    # Update progress
                    my_bar.progress(int((i / len(tickers)) * 100), text=f"Processing {ticker}...")
                    
                    # Filter for specific symbol
                    df_sym = df_all[df_all['symbol'] == ticker].sort_values('timestamp')
                    if df_sym.empty:
                        continue
                        
                    # Extract Today's metrics
                    today_data = df_sym[df_sym['Date'] == latest_date]
                    if today_data.empty:
                        continue
                        
                    current_price = float(today_data['close'].iloc[-1])
                    daily_high = float(today_data['high'].max())
                    daily_low = float(today_data['low'].min())
                    
                    # Distances
                    dist_to_high = current_price - daily_high
                    pct_dist_high = (dist_to_high / daily_high) * 100
                    
                    dist_from_low = current_price - daily_low
                    pct_dist_low = (dist_from_low / daily_low) * 100
                    
                    # Optional: Distance from prior day close
                    prev_close = None
                    pct_change_1d = None
                    if prev_date is not None:
                        prev_data = df_sym[df_sym['Date'] <= prev_date]
                        if not prev_data.empty:
                            prev_close = float(prev_data['close'].iloc[-1])
                            pct_change_1d = ((current_price - prev_close) / prev_close) * 100
                            
                    results.append({
                        "Ticker": ticker,
                        "Live Price": current_price,
                        "Daily High": daily_high,
                        "Dist to High (%)": pct_dist_high,
                        "Daily Low": daily_low,
                        "Dist to Low (%)": pct_dist_low,
                        "1D Change (%)": pct_change_1d if pct_change_1d is not None else 0.0
                    })
                    
                my_bar.progress(100, text="Scan Complete!")
                
                if not results:
                    st.warning("No valid data could be processed for the provided tickers.")
                    st.stop()
                    
                # Convert to DataFrame for display
                res_df = pd.DataFrame(results)
                
                # --- Styling the DataFrame ---
                def color_negative_red(val):
                    color = 'red' if val < 0 else 'green'
                    return f'color: {color}'
                    
                def format_row_bg(row):
                    # Highlight rows near highs (green tint) or near lows (red tint)
                    # For example, if within 0.5% of high:
                    if row['Dist to High (%)'] >= -0.5:
                        return ['background-color: #E8F5E9'] * len(row)
                    elif row['Dist to Low (%)'] <= 0.5:
                        return ['background-color: #FFEBEE'] * len(row)
                    return [''] * len(row)

                formatted_df = res_df.style.format({
                    "Live Price": "${:.2f}",
                    "Daily High": "${:.2f}",
                    "Dist to High (%)": "{:.2f}%",
                    "Daily Low": "${:.2f}",
                    "Dist to Low (%)": "+{:.2f}%",
                    "1D Change (%)": "{:.2f}%"
                }).applymap(color_negative_red, subset=['Dist to High (%)', '1D Change (%)']) \
                  .apply(format_row_bg, axis=1)
                  
                st.dataframe(formatted_df, use_container_width=True, height=500)
                
                st.markdown("""
                <br/>
                **Legend:**  
                - 🟢 Green colored rows indicate an asset trading **within 0.5%** of its Daily High.  
                - 🔴 Red colored rows indicate an asset trading **within 0.5%** of its Daily Low.
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error executing scanner: {e}")

# =====================================================================
# --- Page 7: Signal 5 - PM High / Low Prices ---
# =====================================================================
elif page == "Signal 5 - PM High / Low Prices":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pre-Market Search")
    pm_symbol = st.sidebar.text_input("Ticker", value="QQQ").upper()
    
    st.title(f"🌅 Signal 5: {pm_symbol} Pre-Market Range")
    st.markdown("Isolates today's extended hours trading session (04:00 AM - 09:30 AM EST) to extract the absolute PM High and PM Low.")
    
    if st.button("Calculate PM Range", type="primary"):
        with st.spinner(f"Fetching extended hours data for {pm_symbol}..."):
            try:
                alpaca_key = st.secrets["alpaca"]["API_KEY"]
                alpaca_secret = st.secrets["alpaca"]["API_SECRET"]
                client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
                
                # Fetch 1-minute data for the last few days to make sure we capture today's PM session
                now = datetime.now(timezone.utc)
                start_dt = now - timedelta(days=5) 
                
                req = StockBarsRequest(
                    symbol_or_symbols=pm_symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=start_dt,
                    end=now,
                    feed=DataFeed.IEX # Reverted to IEX due to Alpaca Basic subscription limits on SIP
                )
                
                bars = client.get_stock_bars(req)
                if not bars or bars.df.empty:
                    st.error("No data found.")
                    st.stop()
                    
                df = bars.df.droplevel(0)
                
                # Convert timezone
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                df.index = df.index.tz_convert('America/New_York')
                
                # Get the latest day with data
                df['Date'] = df.index.date
                unique_days = df['Date'].unique()
                latest_day = unique_days[-1]
                
                # Isolate today's data
                df_latest_day = df[df['Date'] == latest_day].copy()
                
                # Filter specifically for Pre-Market window (04:00 AM to 09:30 AM EST)
                pm_data = df_latest_day.between_time('04:00', '09:29')
                
                if pm_data.empty:
                    st.warning(f"No Pre-Market data recorded for {pm_symbol} on {latest_day}.")
                    st.stop()
                    
                # Calculate Absolute High and Low during the PM session from Alpaca (IEX)
                pm_high_alpaca = float(pm_data['high'].max())
                pm_low_alpaca = float(pm_data['low'].min())
                pm_volume = int(pm_data['volume'].sum())
                
                # --- YAHOO FINANCE FALLBACK FOR ACCURATE SIP REPLACEMENT ---
                # Since IEX misses full market wicks, use yfinance to supplement the PM range
                import yfinance as yf
                yf_ticker = yf.Ticker(pm_symbol)
                
                # Fetch 1m data for the last 2 days with prepost=True to get extended hours
                yf_df = yf_ticker.history(period="2d", interval="1m", prepost=True)
                
                pm_high_final = pm_high_alpaca
                pm_low_final = pm_low_alpaca
                
                if not yf_df.empty:
                    # Timezone is usually America/New_York from yfinance
                    yf_df.index = yf_df.index.tz_convert('America/New_York')
                    yf_latest_day = yf_df[yf_df.index.date == latest_day]
                    
                    if not yf_latest_day.empty:
                        yf_pm_data = yf_latest_day.between_time('04:00', '09:29')
                        if not yf_pm_data.empty:
                            yf_pm_high = float(yf_pm_data['High'].max())
                            yf_pm_low = float(yf_pm_data['Low'].min())
                            
                            # Use the extreme from either Alpaca or YF to ensure accuracy
                            pm_high_final = max(pm_high_alpaca, yf_pm_high)
                            pm_low_final = min(pm_low_alpaca, yf_pm_low)
                            
                latest_close = float(df_latest_day['close'].iloc[-1])
                
                st.markdown(f"### Pre-Market Analytics for {latest_day}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Pre-Market High (Consolidated)", f"${pm_high_final:.2f}")
                c2.metric("Pre-Market Low (Consolidated)", f"${pm_low_final:.2f}")
                c3.metric("Total PM Volume (IEX routed)", f"{pm_volume:,}")
                
                st.markdown("---")
                st.markdown(f"**Current Live Price:** ${latest_close:.2f}")
                
                # Determine relationship
                if latest_close > pm_high_final:
                    st.markdown(f'<div class="status-green">🟢 BULLISH: Price (${latest_close:.2f}) is currently trading ABOVE the Pre-Market High (${pm_high_final:.2f}).</div>', unsafe_allow_html=True)
                elif latest_close < pm_low_final:
                    st.markdown(f'<div class="status-red">🔴 BEARISH: Price (${latest_close:.2f}) is currently trading BELOW the Pre-Market Low (${pm_low_final:.2f}).</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-orange">🟡 NEUTRAL: Price (${latest_close:.2f}) is currently trading INSIDE the Pre-Market range (${pm_low_final:.2f} - ${pm_high_final:.2f}).</div>', unsafe_allow_html=True)
                
                # Optional details
                with st.expander("View Raw Pre-Market Time Series"):
                    st.dataframe(pm_data[['open', 'high', 'low', 'close', 'volume']], use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error fetching PM range: {e}")

