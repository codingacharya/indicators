# streamlit_live_trading_alerts_lb_ub.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("Live Stock Dashboard with Buy/Sell Alerts and Upper/Lower Bounds")

# --- User Inputs ---
ticker = st.text_input("Enter Stock Symbol", "AAPL").upper()
interval = st.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
period = st.selectbox("Select Period", ["1d", "5d", "1mo", "3mo"], index=0)
sma_short = st.slider("Short-term SMA Window", 2, 20, 5)
sma_long = st.slider("Long-term SMA Window", 5, 50, 15)
boll_window = st.slider("Bollinger Bands Window", 5, 30, 5)
boll_k = st.slider("Bollinger Bands Std Multiplier", 1, 3, 2)
refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
rsi_period = st.slider("RSI Period", 5, 30, 14)

# --- RSI Function ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Main Loop ---
st.subheader(f"{ticker} Live Data")
placeholder = st.empty()
alert_placeholder = st.empty()  # For alerts

while True:
    # Fetch latest data
    df = yf.download(tickers=ticker, period=period, interval=interval)

    if df.empty:
        placeholder.warning("No data available. Try another symbol or interval.")
        break

    # --- Indicators ---
    df['SMA_short'] = df['Close'].rolling(window=sma_short).mean()
    df['SMA_long'] = df['Close'].rolling(window=sma_long).mean()
    df['RSI'] = compute_rsi(df['Close'], period=rsi_period)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands for LB/UB ---
    df['SMA_BB'] = df['Close'].rolling(window=boll_window).mean()
    df['Std_BB'] = df['Close'].rolling(window=boll_window).std()
    df['UB'] = df['SMA_BB'] + boll_k * df['Std_BB']
    df['LB'] = df['SMA_BB'] - boll_k * df['Std_BB']

    # --- Buy/Sell Signals ---
    df['Buy'] = ((df['SMA_short'] > df['SMA_long']) & (df['RSI'] < 30))
    df['Sell'] = ((df['SMA_short'] < df['SMA_long']) & (df['RSI'] > 70))

    # --- Alert Latest Row ---
    last_price = df['Close'].iloc[-1]
    buy_signal = bool(df['Buy'].iloc[-1])
    sell_signal = bool(df['Sell'].iloc[-1])
    last_time = df.index[-1]

    if buy_signal:
        alert_placeholder.success(
            f"🟢 BUY signal detected for {ticker} at {last_time} (Price: {last_price:.2f})"
        )
    elif sell_signal:
        alert_placeholder.warning(
            f"🔴 SELL signal detected for {ticker} at {last_time} (Price: {last_price:.2f})"
        )
    else:
        alert_placeholder.info("No new signals.")

    with placeholder.container():
        st.write("Latest Data")
        st.dataframe(df.tail(10)[['Close','SMA_short','SMA_long','UB','LB','RSI','MACD','Signal','Buy','Sell']])

        # --- Price Chart with Signals and LB/UB ---
        st.subheader("Price & SMA with Buy/Sell Signals and Bollinger Bands")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax.plot(df.index, df['SMA_short'], label=f'SMA({sma_short})', color='orange')
        ax.plot(df.index, df['SMA_long'], label=f'SMA({sma_long})', color='green')
        ax.plot(df.index, df['UB'], label='Upper Bound (UB)', color='red', linestyle='--')
        ax.plot(df.index, df['LB'], label='Lower Bound (LB)', color='green', linestyle='--')
        ax.scatter(df.index[df['Buy']], df['Close'][df['Buy']], marker='^', color='green', s=100, label='Buy Signal')
        ax.scatter(df.index[df['Sell']], df['Close'][df['Sell']], marker='v', color='red', s=100, label='Sell Signal')
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # --- RSI Plot ---
        st.subheader("RSI")
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_ylabel("RSI")
        ax2.legend()
        st.pyplot(fig2)

        # --- MACD Plot ---
        st.subheader("MACD")
        fig3, ax3 = plt.subplots(figsize=(12, 3))
        ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax3.plot(df.index, df['Signal'], label='Signal', color='red')
        ax3.axhline(0, color='black', linestyle='--')
        ax3.legend()
        st.pyplot(fig3)

    # Wait before refreshing
    time.sleep(refresh_rate)
