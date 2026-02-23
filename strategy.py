import pandas as pd
import numpy as np
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConfluenceVolumeSniper:
    def __init__(self, symbol, account_balance=10000.0, risk_pct=0.01):
        self.symbol = symbol.upper()
        self.account_balance = account_balance
        self.risk_pct = risk_pct
        
    def fetch_data(self, interval='1h', period='730d'):
        """Fetches OHLCV data using yfinance."""
        try:
            # yfinance valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            if interval == '1m':
                period = '7d'
            elif interval in ['2m', '5m', '15m', '30m', '90m']:
                if period in ['90d', '1y', '2y', '5y', '10y', 'max', 'ytd'] or (period.endswith('d') and period[:-1].isdigit() and int(period[:-1]) > 60):
                    period = '60d'
            elif interval in ['60m', '1h']:
                if period in ['5y', '10y', 'max'] or (period.endswith('d') and period[:-1].isdigit() and int(period[:-1]) > 730):
                    period = '2y'

            df = yf.download(self.symbol, interval=interval, period=period, progress=False)
            if df.empty:
                return pd.DataFrame()
            
            # Format columns (yfinance sometimes returns MultiIndex columns if single ticker downloaded)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]
                
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def step1_top_down_analysis(self):
        """
        Step 1: Top-Down Analysis
        Determines Daily trend (UPTREND, DOWNTREND, CHOPPY)
        """
        df_daily = self.fetch_data(interval='1d', period='2y')
        if df_daily.empty or len(df_daily) < 200:
            return "UNKNOWN", df_daily
        
        df_daily['SMA_50'] = df_daily['close'].rolling(50).mean()
        df_daily['SMA_200'] = df_daily['close'].rolling(200).mean()
        
        current_close = float(df_daily['close'].iloc[-1])
        sma50 = float(df_daily['SMA_50'].iloc[-1])
        sma200 = float(df_daily['SMA_200'].iloc[-1])
        
        if pd.isna(sma50) or pd.isna(sma200):
            return "UNKNOWN", df_daily
            
        if current_close > sma50 and sma50 > sma200:
            return "UPTREND", df_daily
        elif current_close < sma50 and sma50 < sma200:
            return "DOWNTREND", df_daily
        else:
            return "CHOPPY", df_daily

    def analyze_operational(self, df_op, trend):
        """
        Steps 2-5: Confluence, PA validation, Volume Validation, & Execution
        """
        if df_op.empty:
            return None, df_op
            
        # Step 2: Confluence zones
        df_op['SMA_8'] = df_op['close'].rolling(window=8).mean()
        df_op['SMA_21'] = df_op['close'].rolling(window=21).mean()
        
        # Dynamic Fibonacci based on 50-period swing high/low lookback
        lookback = 50
        recent_low = df_op['low'].rolling(window=lookback).min()
        recent_high = df_op['high'].rolling(window=lookback).max()
        
        fib_range = recent_high - recent_low
        df_op['Fib_50'] = recent_high - (fib_range * 0.50)
        df_op['Fib_618'] = recent_high - (fib_range * 0.618)
        
        # Step 3: Price Action Signals
        df_op['body'] = abs(df_op['close'] - df_op['open'])
        df_op['lower_tail'] = df_op[['open', 'close']].min(axis=1) - df_op['low']
        df_op['upper_tail'] = df_op['high'] - df_op[['open', 'close']].max(axis=1)
        
        # Pin Bars
        df_op['is_bullish_pinbar'] = (df_op['lower_tail'] > (2 * df_op['body'])) & (df_op['upper_tail'] < df_op['body'])
        df_op['is_bearish_pinbar'] = (df_op['upper_tail'] > (2 * df_op['body'])) & (df_op['lower_tail'] < df_op['body'])
        
        # Engulfing
        df_op['is_bullish_engulfing'] = (df_op['close'] > df_op['open'].shift(1)) & (df_op['open'] <= df_op['close'].shift(1)) & (df_op['close'].shift(1) < df_op['open'].shift(1))
        df_op['is_bearish_engulfing'] = (df_op['close'] < df_op['open'].shift(1)) & (df_op['open'] >= df_op['close'].shift(1)) & (df_op['close'].shift(1) > df_op['open'].shift(1))
        
        # Inside Bar False Breakout
        df_op['is_inside_bar'] = (df_op['high'] < df_op['high'].shift(1)) & (df_op['low'] > df_op['low'].shift(1))
        df_op['is_bullish_false_breakout'] = df_op['is_inside_bar'].shift(1) & (df_op['low'] < df_op['low'].shift(1)) & (df_op['close'] > df_op['high'].shift(1))
        df_op['is_bearish_false_breakout'] = df_op['is_inside_bar'].shift(1) & (df_op['high'] > df_op['high'].shift(1)) & (df_op['close'] < df_op['low'].shift(1))
        
        # Step 4: Validate with Volume (VPA)
        df_op['Vol_SMA'] = df_op['volume'].rolling(window=20).mean()
        df_op['high_volume'] = df_op['volume'] > (df_op['Vol_SMA'] * 1.5)
        
        # Anomaly Filter: High spread, low volume = trap (don't execute)
        atr = (df_op['high'] - df_op['low']).rolling(14).mean()
        massive_spread = (df_op['high'] - df_op['low']) > (atr * 1.5)
        df_op['is_anomaly'] = massive_spread & ~df_op['high_volume']
        
        # Step 5: Trade Execution Check
        last_idx = df_op.index[-1]
        last = df_op.loc[last_idx]
        
        if trend == "CHOPPY" or trend == "UNKNOWN":
            return None, df_op
            
        signal = None
        entry_price = float(last['close'])
        stop_loss = 0.0
        
        # Enforce proximity to confluence Hot Points (within 0.3%)
        near_sma = (abs(entry_price - last['SMA_8']) / entry_price < 0.003) or (abs(entry_price - last['SMA_21']) / entry_price < 0.003)
        near_fib = (abs(entry_price - last['Fib_50']) / entry_price < 0.003) or (abs(entry_price - last['Fib_618']) / entry_price < 0.003)
        
        if not (near_sma or near_fib):
            return None, df_op
            
        # Buy Setup Validation
        if trend == "UPTREND":
            pa_signal = last['is_bullish_pinbar'] or last['is_bullish_engulfing'] or last['is_bullish_false_breakout']
            valid = pa_signal and last['high_volume'] and not last['is_anomaly']
            if valid:
                signal = "BUY"
                stop_loss = float(last['low'] - (entry_price * 0.002)) # ATR buffer
                
        # Sell Setup Validation
        elif trend == "DOWNTREND":
            pa_signal = last['is_bearish_pinbar'] or last['is_bearish_engulfing'] or last['is_bearish_false_breakout']
            valid = pa_signal and last['high_volume'] and not last['is_anomaly']
            if valid:
                signal = "SELL"
                stop_loss = float(last['high'] + (entry_price * 0.002))

        # Output the Trade Plan
        if signal:
            account_risk = self.account_balance * self.risk_pct
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit == 0: 
                return None, df_op
                
            position_size = account_risk / risk_per_unit
            
            if signal == "BUY":
                take_profit_2 = entry_price + (risk_per_unit * 2)
                take_profit_3 = entry_price + (risk_per_unit * 3)
                cons_entry = entry_price - (last['body'] * 0.5)
            else:
                take_profit_2 = entry_price - (risk_per_unit * 2)
                take_profit_3 = entry_price - (risk_per_unit * 3)
                cons_entry = entry_price + (last['body'] * 0.5)
                
            result = {
                "Action": signal,
                "Trend": trend,
                "Aggressive Entry Price": round(entry_price, 2),
                "Conservative Entry (50% Retrace)": round(cons_entry, 2),
                "Stop Loss": round(stop_loss, 2),
                "Risk Per Trade ($)": round(account_risk, 2),
                "Position Size": round(position_size, 4),
                "Target 1 (1:2 RR)": round(take_profit_2, 2),
                "Target 2 (1:3 RR)": round(take_profit_3, 2),
                "Confirmation": "Valid candlestick trigger with volume surge detected at Confluence Zone.",
                "Timestamp": str(last_idx)
            }
            return result, df_op
            
        return None, df_op

    def run(self, interval='1h', period='730d'):
        trend, df_daily = self.step1_top_down_analysis()
        df_op = self.fetch_data(interval, period)
        
        result, df_op = self.analyze_operational(df_op, trend)
        return trend, result, df_daily, df_op
