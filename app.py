import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from data import get_data, get_long_data
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

st.title("AI Trading System")

# session state
if "portfolio" not in st.session_state:
    st.session_state.portfolio = init_portfolio()
if "paper" not in st.session_state:
    st.session_state.paper = init_paper()

# top movers
st.header("Top Movers")
for s, c in get_top_movers():
    st.write(s, "→", round(c, 2), "%")

# ─── INPUT ROW ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    stocks_input = st.text_input("Stocks (comma separated)", "RELIANCE.NS")

with col2:
    timeframe = st.selectbox(
        "⏱ Timeframe",
        ["1 Day (5m)", "5 Days (15m)", "1 Month (1h)", "3 Month (1d)"]
    )

with col3:
    # ✅ FIX: Chart type is NOW OUTSIDE the analyze button — no crash on change
    chart_type = st.selectbox(
        "Chart Type",
        ["Candlestick", "Line", "Bar", "OHLC", "Area"]
    )

with col4:
    mode = st.selectbox("⚙ Mode", ["Advisor", "Paper Trading"])

# ─── ANALYZE BUTTON ─────────────────────────────────────────────────────────
if st.button("Analyze"):
    stocks = [s.strip() for s in stocks_input.split(",") if s.strip()]

    for stock in stocks:
        st.subheader(f"{stock}")

        # ── download data based on timeframe ──
        tf_map = {
            "1 Day (5m)":    ("1d",  "5m"),
            "5 Days (15m)":  ("5d",  "15m"),
            "1 Month (1h)":  ("1mo", "1h"),
            "3 Month (1d)":  ("3mo", "1d"),
        }
        period, interval = tf_map[timeframe]

        df = yf.download(stock, period=period, interval=interval,
                         auto_adjust=True, progress=False)

        # ── clean ──
        from data import clean_df, add_indicators
        df = clean_df(df)

        if df is None:
            st.error(f"Data load failed for {stock}")
            continue

        df = add_indicators(df)

        if len(df) < 20:
            st.warning(f"Not enough data for {stock}")
            continue

        # moving averages (safe - only if enough rows)
        if len(df) >= 20:
            df['ma20'] = df['Close'].rolling(20).mean()
        if len(df) >= 50:
            df['ma50'] = df['Close'].rolling(50).mean()

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
        st.markdown("ML Prediction")
        X, y = prepare_data(df)
        m, acc = train_model(X, y)
        pred = predict(m, df)

        if pred is None:
            st.warning("Prediction not available (not enough indicators)")
        else:
            current_price = float(df['Close'].iloc[-1])
            signal = "BUY" if pred > current_price else "SELL"
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Signal", signal)
            col_b.metric("Current Price", f"₹{current_price:.2f}")
            col_c.metric("Predicted Price", f"₹{pred:.2f}")

        # ─── RISK SCORE ───────────────────────────────────────────────────
        st.write("Risk:", risk_score(df))

        # ─── LSTM ─────────────────────────────────────────────────────────
        st.markdown("LSTM Deep Learning Prediction")
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
        st.markdown("News Sentiment")
        with st.spinner("Fetching latest news..."):
            news = fetch_news(stock)

        res = analyze_news(news)

        for item in res:
            icon = "🟢" if item["sentiment"] == "POSITIVE" else "🔴"
            st.write(f"{icon} **{item['sentiment']}** ({item['score']}) — {item['news']}")

        st.info(f"Overall Sentiment: {overall_sentiment(res)}")

        # ─── BACKTEST ─────────────────────────────────────────────────────
        st.markdown("Backtest")
        if st.button(f"Run Backtest: {stock}", key=f"bt_{stock}"):
            if 'rsi' not in df.columns:
                st.warning("RSI not available for backtest")
            else:
                result = run_backtest(df)
                st.write(result)

        # ─── PAPER TRADING ─────────────────────────────────────────────────
        if mode == "Paper Trading":
            st.markdown("Paper Trading")
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

            st.write("Balance:", f"₹{st.session_state.paper['balance']:,.2f}")
            st.write("Holdings:", st.session_state.paper["shares"])

        st.divider()