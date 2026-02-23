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
    st.sidebar.markdown("---")
    st.sidebar.subheader("Testing Parameters")
    test_symbol = st.sidebar.text_input("Test Ticker", value="QQQ").upper()
    
    st.title(f"🧪 Stock Testing: {test_symbol} Multi-Timeframe Monitor")
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
                        st.info("ℹ️ **Data Source:** Institutional-grade market data provided by the [Alpaca API](https://alpaca.markets/data). Moving averages are calculated locally via Pandas.")

