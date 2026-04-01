import streamlit as st

import plotly.graph_objects as go

from data import get_data, get_long_data, get_live_data, clean_df, add_indicators
from live_price import get_current_price
from model import prepare_data, train_model, predict, risk_score
from lstm_model import prepare_lstm_data, train_lstm, predict_lstm
from news import fetch_news, analyze_news, overall_sentiment
from scanner import get_top_movers
from portfolio import init_portfolio, add_stock, calculate
from paper_trading import init_paper, buy, sell
from backtest import run_backtest

st.set_page_config(layout="wide")

st.markdown("""
<style>
.main { background-color: #0e1117; }
.stButton>button {
    width: 100%;
    border-radius: 10px;
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    height: 45px;
}
.stTextInput>div>div>input { border-radius: 10px; }
.stSelectbox label { font-size: 18px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Trading System")

# session state
if "portfolio" not in st.session_state:
    st.session_state.portfolio = init_portfolio()
if "paper" not in st.session_state:
    st.session_state.paper = init_paper()

# top movers
st.header("🔥 Top Movers")
for s, c in get_top_movers():
    st.write(s, "→", round(c, 2), "%")

# ─── INPUT ROW ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    stocks_input = st.text_input("📈 Stocks (comma separated)", "RELIANCE.NS")

with col2:
    timeframe = st.selectbox(
        "⏱ Timeframe",
        ["1 Day (5m)", "5 Days (15m)", "1 Month (1h)", "3 Month (1d)"]
    )

with col3:
    # ✅ FIX: Chart type is NOW OUTSIDE the analyze button — no crash on change
    chart_type = st.selectbox(
        "📊 Chart Type",
        ["Candlestick", "Line", "Bar", "OHLC", "Area"]
    )

with col4:
    mode = st.selectbox("⚙ Mode", ["Advisor", "Paper Trading"])

# ── AUTO REFRESH CONTROLS ──
col_r1, col_r2 = st.columns([1, 3])
with col_r1:
    auto_refresh = st.toggle("🔄 Auto Refresh Price", value=False)
with col_r2:
    refresh_interval = st.select_slider(
        "Refresh every",
        options=[5, 10, 15, 30, 60],
        value=15,
        format_func=lambda x: f"{x} sec"
    ) if auto_refresh else 15

# ─── ANALYZE BUTTON ─────────────────────────────────────────────────────────
# store analyzed stocks for auto refresh
if "analyzed_stocks" not in st.session_state:
    st.session_state.analyzed_stocks = []

analyze_clicked = st.button("🔍 Analyze")

if analyze_clicked:
    raw = [s.strip() for s in stocks_input.split(",") if s.strip()]
    # normalize symbols
    normalized = []
    for s in raw:
        if s.upper().endswith(".EQ"):
            s = s.replace(".EQ", ".NS").replace(".eq", ".NS")
        if s.upper().endswith(".BSE"):
            s = s.replace(".BSE", ".BO")
        s = s.strip().upper()
        normalized.append(s)
    st.session_state.analyzed_stocks = normalized

if analyze_clicked:
    stocks = st.session_state.analyzed_stocks

    for stock in stocks:
        st.subheader(f"📌 {stock}")

        # ── live data fetch ──
        tf_map = {
            "1 Day (5m)":    ("1d",  "5m"),
            "5 Days (15m)":  ("5d",  "15m"),
            "1 Month (1h)":  ("1mo", "1h"),
            "3 Month (1d)":  ("3mo", "1d"),
        }
        period, interval = tf_map[timeframe]

        with st.spinner(f"Fetching live data for {stock}..."):
            df, source = get_live_data(stock, period=period, interval=interval)

        if df is None:
            st.error(f"❌ Data load failed for {stock}")
            continue

        # ✅ source badge — user ko pata chale live hai ya delayed
        st.info("🟡 Chart data (15min delayed) — yfinance")

        # ── LIVE PRICE (NSE real-time with auto refresh) ──
        st.markdown("### 💰 Live Price")

        # placeholder — auto refresh pe update hoga in-place
        price_placeholder = st.empty()

        def show_price(sym):
            live = get_current_price(sym)
            with price_placeholder.container():
                if live:
                    arrow  = "▲" if live["pChange"] >= 0 else "▼"
                    status = "🟢 Market Open — Live" if live["market_open"] else "🔴 Market Closed — Last Close"
                    src    = "NSE Live" if live.get("source") == "nse_live" else "yfinance"
                    st.caption(f"{status}  ·  Source: {src}")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric(
                        "Price", f"₹{live['price']:,.2f}",
                        f"{arrow} {live['pChange']:.2f}% ({live['change']:.2f})"
                    )
                    c2.metric("Open",       f"₹{live['open']:,.2f}")
                    c3.metric("Day High",   f"₹{live['high']:,.2f}")
                    c4.metric("Day Low",    f"₹{live['low']:,.2f}")
                    c5.metric("Prev Close", f"₹{live['close']:,.2f}")
                    st.caption(f"Last updated: {live['time']} IST")
                else:
                    from live_price import is_market_open
                    if not is_market_open():
                        st.info("🔴 Market closed — Live price 9:15 AM se 3:30 PM IST milega")
                    else:
                        st.warning("⚠ NSE se data nahi aaya — internet check karo")

        # first fetch
        show_price(stock)

        if len(df) < 10:
            st.warning(f"⚠ Not enough data for {stock} — try a larger timeframe (5 Days ya 1 Month)")
            continue

        # ma20/ma50 already added by add_indicators inside get_live_data

        # ─── CHART ──────────────────────────────────────────────────────────
        from plotly.subplots import make_subplots

        # Volume ke liye color (green = up day, red = down day)
        if 'Volume' in df.columns:
            vol_colors = [
                '#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i]
                else '#ef5350'
                for i in range(len(df))
            ]
            has_volume = True
        else:
            has_volume = False

        # 2 rows — row1: price, row2: volume (agar available ho)
        if has_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.75, 0.25],
                vertical_spacing=0.02
            )
        else:
            fig = make_subplots(rows=1, cols=1)

        # ── Price chart (row 1) ──
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ), row=1, col=1)

        elif chart_type == "OHLC":
            fig.add_trace(go.Ohlc(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price"
            ), row=1, col=1)

        elif chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name="Close",
                line=dict(color='#00bfff', width=2)
            ), row=1, col=1)

        elif chart_type == "Bar":
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Close'],
                name="Close",
                marker_color='#7e57c2'
            ), row=1, col=1)

        elif chart_type == "Area":
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name="Close",
                fill='tozeroy',
                line=dict(color='#26a69a', width=2),
                fillcolor='rgba(38,166,154,0.2)'
            ), row=1, col=1)

        # ── Moving averages overlay ──
        if 'ma20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ma20'],
                line=dict(color='#ffa726', width=1.5),
                name="MA20"
            ), row=1, col=1)

        if 'ma50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ma50'],
                line=dict(color='#ef5350', width=1.5),
                name="MA50"
            ), row=1, col=1)

        # ── Volume chart (row 2) ──
        if has_volume:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.7
            ), row=2, col=1)

        # ── Layout ──
        xaxis_type = 'category' if interval in ['5m', '15m', '1h'] else 'date'

        fig.update_layout(
            template="plotly_dark",
            height=650,
            title=f"{stock} — {timeframe} ({chart_type})",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            margin=dict(l=10, r=10, t=50, b=60)
        )

        # shared x-axis settings
        fig.update_xaxes(type=xaxis_type, nticks=30, tickangle=-45)

        # y-axis labels
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        if has_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # ─── ML PREDICTION ───────────────────────────────────────────────────
        st.markdown("### 🤖 ML Prediction")
        X, y = prepare_data(df)
        m, acc = train_model(X, y)
        pred = predict(m, df)

        if pred is None:
            st.warning("Prediction not available (not enough indicators)")
        else:
            current_price = float(df['Close'].iloc[-1])
            signal = "BUY 📈" if pred > current_price else "SELL 📉"
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Signal", signal)
            col_b.metric("Current Price", f"₹{current_price:.2f}")
            col_c.metric("Predicted Price", f"₹{pred:.2f}")

        # ─── RISK SCORE ───────────────────────────────────────────────────
        st.write("📊 Risk:", risk_score(df))

        # ─── LSTM ─────────────────────────────────────────────────────────
        st.markdown("### 🧠 LSTM Deep Learning Prediction")
        df_long = get_long_data(stock)

        if df_long is None or len(df_long) < 60:
            st.warning("Not enough historical data for LSTM (need 60+ days)")
        else:
            with st.spinner("Training LSTM..."):
                Xl, yl, scaler = prepare_lstm_data(df_long)
                lstm = train_lstm(Xl, yl)
                lstm_price = predict_lstm(lstm, df_long, scaler)
            st.success(f"LSTM Predicted Next Price: ₹{round(float(lstm_price), 2)}")

        # ─── NEWS SENTIMENT ────────────────────────────────────────────────
        st.markdown("### 📰 News Sentiment")
        with st.spinner("Fetching latest news..."):
            news = fetch_news(stock)

        res = analyze_news(news)

        for item in res:
            icon = "🟢" if item["sentiment"] == "POSITIVE" else "🔴"
            st.write(f"{icon} **{item['sentiment']}** ({item['score']}) — {item['news']}")

        st.info(f"Overall Sentiment: {overall_sentiment(res)}")

        # ─── BACKTEST ─────────────────────────────────────────────────────
        st.markdown("### 📉 Backtest")
        if st.button(f"Run Backtest: {stock}", key=f"bt_{stock}"):
            if 'rsi' not in df.columns:
                st.warning("RSI not available for backtest")
            else:
                result = run_backtest(df)
                st.write(result)

        # ─── PAPER TRADING ─────────────────────────────────────────────────
        if mode == "Paper Trading":
            st.markdown("### 💹 Paper Trading")
            price = float(df['Close'].iloc[-1])
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"🟢 Buy {stock}", key=f"buy_{stock}"):
                    st.session_state.paper = buy(st.session_state.paper, stock, price)
                    st.success(f"Bought {stock} @ ₹{price:.2f}")
            with c2:
                if st.button(f"🔴 Sell {stock}", key=f"sell_{stock}"):
                    st.session_state.paper = sell(st.session_state.paper, stock, price)
                    st.success(f"Sold {stock} @ ₹{price:.2f}")

            st.write("💰 Balance:", f"₹{st.session_state.paper['balance']:,.2f}")
            st.write("📦 Holdings:", st.session_state.paper["shares"])

        st.divider()

# ── AUTO REFRESH (price only, outside analyze block) ──
import time as _time
from live_price import is_market_open as _market_open

if auto_refresh and st.session_state.get("analyzed_stocks"):
    st.divider()
    
    # Only refresh during market hours
    if _market_open():
        countdown_placeholder = st.empty()
        
        for i in range(refresh_interval, 0, -1):
            countdown_placeholder.caption(f"🔄 Live price refresh in {i}s... (Auto Refresh ON)")
            _time.sleep(1)
        
        countdown_placeholder.empty()
        
        # Show updated prices for all analyzed stocks
        st.markdown("### 💰 Auto-Refreshed Prices")
        for sym in st.session_state.analyzed_stocks:
            live = get_current_price(sym)
            if live:
                arrow = "▲" if live["pChange"] >= 0 else "▼"
                col1, col2, col3 = st.columns(3)
                col1.metric(sym, f"₹{live['price']:,.2f}", 
                           f"{arrow} {live['pChange']:.2f}%")
                col2.metric("High", f"₹{live['high']:,.2f}")
                col3.metric("Low",  f"₹{live['low']:,.2f}")
        
        # Rerun to refresh again
        st.rerun()
    else:
        st.info("🔴 Market closed — Auto refresh sirf 9:15 AM - 3:30 PM kaam karta hai")
