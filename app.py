import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time as _time
import pytz
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('brown', quiet=True)
except Exception:
    pass

from data import get_data, get_long_data, get_live_data, clean_df, add_indicators
from live_price import get_current_price, is_market_open
from model import prepare_data, train_model, predict, risk_score
from lstm_model import prepare_lstm_data, train_lstm, predict_lstm
from news import fetch_news, analyze_news, overall_sentiment
from scanner import get_top_movers
from portfolio import init_portfolio, add_stock, calculate
from paper_trading import init_paper, buy, sell
from backtest import run_backtest

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="QuantEdge — AI Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080b10;
    color: #e8edf5;
}

.main .block-container {
    padding: 0 2rem 2rem 2rem;
    max-width: 1400px;
}

/* ── HIDE STREAMLIT DEFAULTS ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── TOP BAR ── */
.top-bar {
    background: rgba(8,11,16,0.95);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 14px 0 10px 0;
    margin-bottom: 24px;
}

.top-bar-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #e8edf5;
    letter-spacing: -0.5px;
}

.logo-text span { color: #00e5a0; }

/* ── TICKER ── */
.ticker-wrap {
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 8px 16px;
    overflow: hidden;
    margin-bottom: 20px;
}

.ticker-inner {
    display: flex;
    gap: 32px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    overflow-x: auto;
    scrollbar-width: none;
}

.ticker-inner::-webkit-scrollbar { display: none; }

.t-sym { color: #5a6880; }
.t-price { color: #e8edf5; font-weight: 700; }
.t-up { color: #00e5a0; }
.t-dn { color: #ff4560; }

/* ── SECTION CARDS ── */
.card {
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #5a6880;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-title::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00e5a0;
}

/* ── LIVE PRICE CARD ── */
.price-main {
    font-family: 'Syne', sans-serif;
    font-size: 40px;
    font-weight: 800;
    color: #e8edf5;
    letter-spacing: -1px;
    line-height: 1;
}

.price-change-up {
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    color: #00e5a0;
    margin-top: 4px;
}

.price-change-dn {
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    color: #ff4560;
    margin-top: 4px;
}

.price-meta {
    font-size: 11px;
    color: #5a6880;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
}

.stat-mini {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}

.stat-mini-label {
    font-size: 10px;
    color: #5a6880;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
}

.stat-mini-val {
    font-family: 'Space Mono', monospace;
    font-size: 16px;
    font-weight: 700;
    color: #e8edf5;
    margin-top: 4px;
}

/* ── SIGNAL BOX ── */
.signal-buy {
    background: rgba(0,229,160,0.08);
    border: 1px solid rgba(0,229,160,0.25);
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}

.signal-sell {
    background: rgba(255,69,96,0.08);
    border: 1px solid rgba(255,69,96,0.25);
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}

.signal-text-buy {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #00e5a0;
}

.signal-text-sell {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #ff4560;
}

.signal-sub {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #5a6880;
    margin-top: 6px;
}

/* ── NEWS ITEMS ── */
.news-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.news-row:last-child { border-bottom: none; }

.news-dot-pos { width:8px;height:8px;border-radius:50%;background:#00e5a0;flex-shrink:0;margin-top:5px; }
.news-dot-neg { width:8px;height:8px;border-radius:50%;background:#ff4560;flex-shrink:0;margin-top:5px; }
.news-dot-neu { width:8px;height:8px;border-radius:50%;background:#5a6880;flex-shrink:0;margin-top:5px; }

.news-sentiment-pos { font-family:'Space Mono',monospace;font-size:10px;color:#00e5a0;font-weight:700; }
.news-sentiment-neg { font-family:'Space Mono',monospace;font-size:10px;color:#ff4560;font-weight:700; }
.news-sentiment-neu { font-family:'Space Mono',monospace;font-size:10px;color:#5a6880;font-weight:700; }
.news-headline { font-size:13px;color:#9aa5b8;line-height:1.5;margin-top:2px; }

/* ── BUTTONS ── */
.stButton > button {
    background: #00e5a0 !important;
    color: #080b10 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    height: 44px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #111820 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8edf5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: #111820;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px 16px !important;
}

[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: #5a6880 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    color: #e8edf5 !important;
}

/* ── DIVIDER ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── TOGGLE ── */
.stToggle { color: #e8edf5 !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111820;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.07);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    color: #5a6880;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
}

.stTabs [aria-selected="true"] {
    background: #00e5a0 !important;
    color: #080b10 !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #00e5a0 !important; }

/* ── INFO/SUCCESS/WARNING ── */
.stAlert {
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)



# ── SESSION STATE ──
if "portfolio" not in st.session_state:
    st.session_state.portfolio = init_portfolio()
if "paper" not in st.session_state:
    st.session_state.paper = init_paper()
if "analyzed_stocks" not in st.session_state:
    st.session_state.analyzed_stocks = []
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}  # stock → {df, pred, lstm, news, res}
if "price_refresh_only" not in st.session_state:
    st.session_state.price_refresh_only = False
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
if "compare_stock" not in st.session_state:
    st.session_state.compare_stock = ""

# ── TOP BAR ──
st.markdown("""
<div class="top-bar">
    <div class="top-bar-inner">
        <div class="logo-text">Quant<span>Edge</span></div>
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:#5a6880">
            AI Trading Intelligence · NSE/BSE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── LIVE TICKER ──
movers = get_top_movers()
if movers:
    ticker_html = '<div class="ticker-wrap"><div class="ticker-inner">'
    for s, c in movers:
        cls = "t-up" if c >= 0 else "t-dn"
        arrow = "▲" if c >= 0 else "▼"
        ticker_html += f'<span><span class="t-sym">{s}</span> &nbsp;<span class="t-price">—</span>&nbsp;<span class="{cls}">{arrow} {abs(c):.2f}%</span></span>'
    ticker_html += '</div></div>'
    st.markdown(ticker_html, unsafe_allow_html=True)

# ── SIDEBAR: NSE Calendar + Multi-timeframe toggle ──
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:18px;
                color:#e8edf5;margin-bottom:20px;">⚙ Settings</div>
    """, unsafe_allow_html=True)

    # Multi-timeframe toggle
    st.markdown("### 📊 Multi-Timeframe")
    multi_tf = st.toggle("Show 4 timeframes at once", value=False,
                          help="Ek stock ke 4 timeframes ek saath dikhao")
    st.caption("Analyze mein 1 stock honi chahiye")

    st.divider()

    # NSE Calendar
    st.markdown("### 📅 NSE Events Calendar")

    import datetime as _dt
    _today = _dt.date.today()
    _month = _today.month
    _year  = _today.year

    # Key NSE events (static + computed)
    NSE_EVENTS = [
        # Monthly expiry — last Thursday of month
        # Q results season — Jan, Apr, Jul, Oct
        {"date": "2025-04-24", "event": "Monthly F&O Expiry", "type": "expiry"},
        {"date": "2025-04-30", "event": "Sensex Rebalancing", "type": "index"},
        {"date": "2025-05-29", "event": "Monthly F&O Expiry", "type": "expiry"},
        {"date": "2025-06-26", "event": "Monthly F&O Expiry", "type": "expiry"},
        {"date": "2025-07-31", "event": "Monthly F&O Expiry", "type": "expiry"},
        {"date": "2026-01-29", "event": "Monthly F&O Expiry", "type": "expiry"},
        {"date": "2026-02-26", "event": "Monthly F&O Expiry", "type": "expiry"},
        {"date": "2026-03-26", "event": "Monthly F&O Expiry", "type": "expiry"},
        {"date": "2026-04-23", "event": "Monthly F&O Expiry", "type": "expiry"},
        {"date": "2026-04-01", "event": "Q4 Results Season Start", "type": "results"},
        {"date": "2026-07-01", "event": "Q1 Results Season Start", "type": "results"},
        {"date": "2026-10-01", "event": "Q2 Results Season Start", "type": "results"},
        {"date": "2026-01-01", "event": "Q3 Results Season Start", "type": "results"},
        {"date": "2026-02-01", "event": "Union Budget 2026", "type": "budget"},
        {"date": "2025-12-25", "event": "Christmas Holiday", "type": "holiday"},
        {"date": "2026-01-26", "event": "Republic Day Holiday", "type": "holiday"},
        {"date": "2026-03-18", "event": "Holi Holiday", "type": "holiday"},
        {"date": "2026-04-14", "event": "Dr. Ambedkar Jayanti", "type": "holiday"},
    ]

    _type_color = {
        "expiry":  "#ffa726",
        "results": "#00e5a0",
        "budget":  "#7c4dff",
        "holiday": "#ff4560",
        "index":   "#00bfff",
    }
    _type_icon = {
        "expiry":  "⚡",
        "results": "📊",
        "budget":  "🏛",
        "holiday": "🎉",
        "index":   "📈",
    }

    # Show upcoming 6 events
    _upcoming = []
    for ev in NSE_EVENTS:
        try:
            _ev_date = _dt.date.fromisoformat(ev["date"])
            if _ev_date >= _today:
                _days_left = (_ev_date - _today).days
                _upcoming.append({**ev, "date_obj": _ev_date, "days": _days_left})
        except Exception:
            pass

    _upcoming = sorted(_upcoming, key=lambda x: x["days"])[:8]

    for ev in _upcoming:
        _col = _type_color.get(ev["type"], "#5a6880")
        _icon = _type_icon.get(ev["type"], "•")
        _days = ev["days"]
        _days_txt = "Today!" if _days == 0 else f"in {_days}d"
        st.markdown(
            f'<div style="border-left:3px solid {_col};padding:8px 10px;'
            f'margin-bottom:8px;background:rgba(255,255,255,0.02);border-radius:0 6px 6px 0;">'
            f'<div style="font-family:Space Mono,monospace;font-size:10px;color:{_col}">'
            f'{_icon} {ev["date"]} · {_days_txt}</div>'
            f'<div style="font-size:12px;color:#e8edf5;margin-top:2px;">{ev["event"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.caption("⚡ Expiry · 📊 Results · 🏛 Budget · 🎉 Holiday")

# ── WATCHLIST DASHBOARD ──
with st.expander("📋 Watchlist", expanded=True):
    wl_col1, wl_col2 = st.columns([4, 1])
    with wl_col1:
        wl_input = st.text_input("Add stock to watchlist", "", 
                                  placeholder="e.g. SBIN.NS",
                                  label_visibility="collapsed",
                                  key="wl_input")
    with wl_col2:
        if st.button("+ Add", key="wl_add"):
            s = wl_input.strip().upper()
            if s and s not in st.session_state.watchlist:
                st.session_state.watchlist.append(s)

    if st.session_state.watchlist:
        wl_cols = st.columns(len(st.session_state.watchlist))
        for i, sym in enumerate(st.session_state.watchlist):
            with wl_cols[i]:
                live = get_current_price(sym)
                if live:
                    arrow = "▲" if live["pChange"] >= 0 else "▼"
                    col = "#00e5a0" if live["pChange"] >= 0 else "#ff4560"
                    st.markdown(
                        f'<div class="card" style="text-align:center;padding:12px 8px;">'
                        f'<div style="font-family:Space Mono,monospace;font-size:10px;color:#5a6880">{sym.replace(".NS","")}</div>'
                        f'<div style="font-family:Syne,sans-serif;font-size:18px;font-weight:800;margin:4px 0">&#8377;{live["price"]:,.2f}</div>'
                        f'<div style="font-family:Space Mono,monospace;font-size:11px;color:{col}">{arrow} {live["pChange"]:.2f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="card" style="text-align:center;padding:12px 8px;">'
                        f'<div style="font-family:Space Mono,monospace;font-size:10px;color:#5a6880">{sym.replace(".NS","")}</div>'
                        f'<div style="font-size:12px;color:#5a6880;margin-top:6px;">N/A</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                if st.button("✕", key=f"wl_rm_{sym}", help=f"Remove {sym}"):
                    st.session_state.watchlist.remove(sym)
                    st.rerun()

# ── CONTROL PANEL ──
with st.container():
    st.markdown('<div class="card"><div class="card-title">Market Scanner</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 1])
    
    with c1:
        stocks_input = st.text_input("Stock Symbol", "RELIANCE.NS", 
                                      placeholder="e.g. RELIANCE.NS, TCS.NS",
                                      label_visibility="collapsed")
    with c2:
        timeframe = st.selectbox("Timeframe", 
                                  ["1 Day (5m)", "5 Days (15m)", "1 Month (1h)", "3 Month (1d)"],
                                  label_visibility="collapsed")
    with c3:
        chart_type = st.selectbox("Chart", 
                                   ["Candlestick", "Line", "OHLC", "Bar", "Area"],
                                   label_visibility="collapsed")
    with c4:
        mode = st.selectbox("Mode", ["Advisor", "Paper Trading"],
                             label_visibility="collapsed")
    with c5:
        analyze_clicked = st.button("Analyze →")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Auto refresh
col_r1, col_r2 = st.columns([1, 4])
with col_r1:
    auto_refresh = st.toggle("🔄 Auto Refresh", value=False)
with col_r2:
    refresh_interval = st.select_slider(
        "Interval",
        options=[5, 10, 15, 30, 60],
        value=15,
        format_func=lambda x: f"{x}s",
        label_visibility="collapsed"
    ) if auto_refresh else 15

# ── ANALYZE ──
if analyze_clicked:
    raw = [s.strip() for s in stocks_input.split(",") if s.strip()]
    normalized = []
    for s in raw:
        if s.upper().endswith(".EQ"): s = s.replace(".EQ", ".NS")
        if s.upper().endswith(".BSE"): s = s.replace(".BSE", ".BO")
        normalized.append(s.strip().upper())
    st.session_state.analyzed_stocks = normalized
    st.session_state.analysis_done = True  # ✅ analysis complete flag

tf_map = {
    "1 Day (5m)":   ("1d",  "5m"),
    "5 Days (15m)": ("5d",  "15m"),
    "1 Month (1h)": ("1mo", "1h"),
    "3 Month (1d)": ("3mo", "1d"),
}
period, interval = tf_map[timeframe]

# Run full analysis only on first click OR if not price_refresh_only
run_full = analyze_clicked or (
    st.session_state.get("price_refresh_only") and 
    st.session_state.get("analyzed_stocks")
)

# Show analysis if: first click, price refresh, OR any button inside (backtest, buy, sell etc)
_show_analysis = (
    analyze_clicked or
    st.session_state.get("price_refresh_only", False) or
    st.session_state.get("analysis_done", False)
)

if _show_analysis and st.session_state.analyzed_stocks:
    is_refresh = not analyze_clicked and st.session_state.get("analysis_done", False)

    # ── MULTI-TIMEFRAME MODE ──
    if multi_tf and len(st.session_state.analyzed_stocks) == 1:
        _mtf_stock = st.session_state.analyzed_stocks[0]
        st.markdown(f"""
        <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;
                    color:#e8edf5;margin:24px 0 16px 0;">
            📊 {_mtf_stock} — Multi-Timeframe View
        </div>
        """, unsafe_allow_html=True)

        _tf_list = [
            ("1 Day",   "1d",  "5m"),
            ("5 Days",  "5d",  "15m"),
            ("1 Month", "1mo", "1h"),
            ("3 Month", "3mo", "1d"),
        ]
        _mtf_cols = st.columns(2)
        for _idx, (_label, _per, _ivl) in enumerate(_tf_list):
            with _mtf_cols[_idx % 2]:
                with st.spinner(f"{_label}..."):
                    _df_mtf, _ = get_live_data(_mtf_stock, period=_per, interval=_ivl)

                if _df_mtf is not None and len(_df_mtf) > 5:
                    try:
                        _ist = pytz.timezone('Asia/Kolkata')
                        if _df_mtf.index.tzinfo is None:
                            _df_mtf.index = _df_mtf.index.tz_localize('UTC').tz_convert(_ist)
                        else:
                            _df_mtf.index = _df_mtf.index.tz_convert(_ist)
                        _df_mtf.index = _df_mtf.index.tz_localize(None)
                    except Exception:
                        pass

                    _fig_mtf = go.Figure()
                    _is_up_mtf = float(_df_mtf['Close'].squeeze().iloc[-1]) >= float(_df_mtf['Close'].squeeze().iloc[0])
                    _c_mtf = '#00e5a0' if _is_up_mtf else '#ff4560'

                    _fig_mtf.add_trace(go.Candlestick(
                        x=_df_mtf.index,
                        open=_df_mtf['Open'].squeeze(),
                        high=_df_mtf['High'].squeeze(),
                        low=_df_mtf['Low'].squeeze(),
                        close=_df_mtf['Close'].squeeze(),
                        name=_label,
                        increasing=dict(line=dict(color='#00e5a0'), fillcolor='#00e5a0'),
                        decreasing=dict(line=dict(color='#ff4560'), fillcolor='#ff4560'),
                    ))

                    _fig_mtf.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='#0d1117',
                        plot_bgcolor='#0d1117',
                        height=280,
                        title=dict(text=f"{_label}", font=dict(size=13, color='#9aa5b8')),
                        margin=dict(l=0, r=0, t=36, b=0),
                        xaxis_rangeslider_visible=False,
                        showlegend=False,
                        font=dict(family='DM Sans', color='#9aa5b8'),
                    )
                    _xtype = 'category' if _ivl in ['5m', '15m'] else 'date'
                    _fig_mtf.update_xaxes(type=_xtype, nticks=8, tickfont=dict(size=8),
                                          gridcolor='rgba(255,255,255,0.03)')
                    _fig_mtf.update_yaxes(gridcolor='rgba(255,255,255,0.03)',
                                          tickfont=dict(size=8))
                    st.plotly_chart(_fig_mtf, use_container_width=True)

                    # Quick stats
                    _ret = (float(_df_mtf['Close'].squeeze().iloc[-1]) -
                            float(_df_mtf['Close'].squeeze().iloc[0])) / float(_df_mtf['Close'].squeeze().iloc[0]) * 100
                    st.markdown(
                        f'<div style="font-family:Space Mono,monospace;font-size:11px;'
                        f'color:{"#00e5a0" if _ret>=0 else "#ff4560"};text-align:center;'
                        f'margin-bottom:16px;">{"▲" if _ret>=0 else "▼"} {abs(_ret):.2f}% this period</div>',
                        unsafe_allow_html=True
                    )
        st.divider()

    for stock in st.session_state.analyzed_stocks:

        # ── STOCK HEADER ──
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin:24px 0 16px 0;">
            <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:#e8edf5;">{stock}</div>
            <div style="font-family:'Space Mono',monospace;font-size:11px;color:#5a6880;background:#111820;
                        border:1px solid rgba(255,255,255,0.07);padding:4px 10px;border-radius:6px;">NSE · {timeframe}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── FETCH DATA — use cache on non-analyze reruns ──
        cached_df = st.session_state.analysis_cache.get(stock, {}).get('df')

        if analyze_clicked or cached_df is None:
            # Fresh fetch only on Analyze click or first time
            with st.spinner(f"Fetching {stock}..."):
                df, source = get_live_data(stock, period=period, interval=interval)
            if df is not None:
                if stock not in st.session_state.analysis_cache:
                    st.session_state.analysis_cache[stock] = {}
                st.session_state.analysis_cache[stock]['df']       = df
                st.session_state.analysis_cache[stock]['interval'] = interval
        else:
            # Use cached data — no fetch needed (backtest/buy/sell button click)
            df = cached_df

        if df is None:
            st.error(f"❌ Data unavailable for {stock}")
            continue

        if len(df) < 20:
            st.warning(f"⚠ Not enough data — try 5 Days or 1 Month timeframe")
            continue

        # IST fix
        try:
            ist = pytz.timezone('Asia/Kolkata')
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize('UTC').tz_convert(ist)
            else:
                df.index = df.index.tz_convert(ist)
            df.index = df.index.tz_localize(None)
        except Exception:
            pass

        # ── TWO COLUMN LAYOUT ──
        left_col, right_col = st.columns([2, 1])

        with left_col:
            # ── CHART ──
            has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
            if has_volume:
                vol_colors = [
                    '#26a69a' if float(df['Close'].iloc[i]) >= float(df['Open'].iloc[i])
                    else '#ef5350' for i in range(len(df))
                ]
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.78, 0.22], vertical_spacing=0.02)
            else:
                fig = make_subplots(rows=1, cols=1)

            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'].squeeze(),
                    high=df['High'].squeeze(), low=df['Low'].squeeze(),
                    close=df['Close'].squeeze(), name="Price",
                    increasing=dict(line=dict(color='#26a69a'), fillcolor='#26a69a'),
                    decreasing=dict(line=dict(color='#ef5350'), fillcolor='#ef5350'),
                ), row=1, col=1)
            elif chart_type == "OHLC":
                fig.add_trace(go.Ohlc(
                    x=df.index, open=df['Open'].squeeze(),
                    high=df['High'].squeeze(), low=df['Low'].squeeze(),
                    close=df['Close'].squeeze(), name="Price",
                    increasing=dict(line=dict(color='#26a69a')),
                    decreasing=dict(line=dict(color='#ef5350')),
                ), row=1, col=1)
            elif chart_type == "Line":
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'].squeeze(),
                    mode='lines', name="Close",
                    line=dict(color='#00e5a0', width=2)
                ), row=1, col=1)
            elif chart_type == "Bar":
                bar_colors = [
                    '#26a69a' if float(df['Close'].iloc[i]) >= float(df['Open'].iloc[i])
                    else '#ef5350' for i in range(len(df))
                ]
                fig.add_trace(go.Bar(
                    x=df.index, y=df['Close'].squeeze(),
                    name="Close", marker_color=bar_colors, opacity=0.85
                ), row=1, col=1)
            elif chart_type == "Area":
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'].squeeze(),
                    mode='lines', fill='tozeroy', name="Close",
                    line=dict(color='#00e5a0', width=2),
                    fillcolor='rgba(0,229,160,0.1)'
                ), row=1, col=1)

            if 'ma20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['ma20'].squeeze(),
                    mode='lines', line=dict(color='#ffa726', width=1.2, dash='dot'),
                    name="MA20", opacity=0.8
                ), row=1, col=1)

            if 'ma50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['ma50'].squeeze(),
                    mode='lines', line=dict(color='#ef5350', width=1.2, dash='dot'),
                    name="MA50", opacity=0.8
                ), row=1, col=1)

            # ── SUPPORT & RESISTANCE auto-detect ──
            try:
                _close = df['Close'].squeeze()
                _high  = df['High'].squeeze()  if 'High'  in df.columns else _close
                _low   = df['Low'].squeeze()   if 'Low'   in df.columns else _close
                _n     = min(len(df), 60)

                _supports, _resistances = [], []
                for i in range(2, _n - 2):
                    if (_low.iloc[i] < _low.iloc[i-1] and _low.iloc[i] < _low.iloc[i-2] and
                        _low.iloc[i] < _low.iloc[i+1] and _low.iloc[i] < _low.iloc[i+2]):
                        _supports.append(float(_low.iloc[i]))
                    if (_high.iloc[i] > _high.iloc[i-1] and _high.iloc[i] > _high.iloc[i-2] and
                        _high.iloc[i] > _high.iloc[i+1] and _high.iloc[i] > _high.iloc[i+2]):
                        _resistances.append(float(_high.iloc[i]))

                def _cluster(levels, tol=0.005):
                    if not levels: return []
                    levels = sorted(set(levels))
                    clustered, group = [], [levels[0]]
                    for lv in levels[1:]:
                        if (lv - group[-1]) / group[-1] < tol:
                            group.append(lv)
                        else:
                            clustered.append(sum(group)/len(group))
                            group = [lv]
                    clustered.append(sum(group)/len(group))
                    return clustered[-3:]

                for _lv in _cluster(_supports):
                    fig.add_hline(y=_lv, line_dash="dash", line_color="#00e5a0",
                                  line_width=1, opacity=0.5,
                                  annotation_text=f"S ₹{_lv:,.0f}",
                                  annotation_position="left",
                                  annotation_font=dict(size=10, color="#00e5a0"),
                                  row=1, col=1)

                for _lv in _cluster(_resistances):
                    fig.add_hline(y=_lv, line_dash="dash", line_color="#ff4560",
                                  line_width=1, opacity=0.5,
                                  annotation_text=f"R ₹{_lv:,.0f}",
                                  annotation_position="left",
                                  annotation_font=dict(size=10, color="#ff4560"),
                                  row=1, col=1)
            except Exception as _e:
                print(f"[S/R] {_e}")

            if has_volume:
                fig.add_trace(go.Bar(
                    x=df.index, y=df['Volume'].squeeze(),
                    name="Vol", marker_color=vol_colors,
                    opacity=0.5, showlegend=False
                ), row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    font=dict(size=11, family='Space Mono'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                hovermode='x unified',
                font=dict(family='DM Sans', color='#9aa5b8'),
            )

            xtype = 'category' if interval in ['5m', '15m'] else 'date'
            fig.update_xaxes(type=xtype, nticks=20, tickangle=-45,
                            tickfont=dict(size=9), gridcolor='rgba(255,255,255,0.04)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)',
                            tickfont=dict(size=9))

            st.plotly_chart(fig, use_container_width=True)

        # On refresh — use cached ML results
        cache = st.session_state.analysis_cache.get(stock, {})
        
        with right_col:
            # ── LIVE PRICE ──
            live = get_current_price(stock)
            if live:
                arrow = "▲" if live["pChange"] >= 0 else "▼"
                chg_class = "price-change-up" if live["pChange"] >= 0 else "price-change-dn"
                mkt = "🟢 Live" if live["market_open"] else "🔴 Closed"
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">Live Price · {mkt}</div>
                    <div class="price-main">₹{live['price']:,.2f}</div>
                    <div class="{chg_class}">{arrow} {live['pChange']:.2f}% &nbsp;({live['change']:+.2f})</div>
                    <div class="price-meta">Updated {live['time']} IST</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:16px;">
                        <div class="stat-mini"><div class="stat-mini-label">Open</div><div class="stat-mini-val">₹{live['open']:,.0f}</div></div>
                        <div class="stat-mini"><div class="stat-mini-label">Prev Close</div><div class="stat-mini-val">₹{live['close']:,.0f}</div></div>
                        <div class="stat-mini"><div class="stat-mini-label">High</div><div class="stat-mini-val" style="color:#00e5a0">₹{live['high']:,.0f}</div></div>
                        <div class="stat-mini"><div class="stat-mini-label">Low</div><div class="stat-mini-val" style="color:#ff4560">₹{live['low']:,.0f}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">Live Price</div>
                    <div style="color:#5a6880;font-family:'Space Mono',monospace;font-size:12px;">
                        {'🔴 Market closed (9:15–3:30 IST)' if not is_market_open() else '⚠ NSE unreachable'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── RSI CARD with explanation ──
            rsi_val = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns and not df['rsi'].isna().all() else None
            risk    = risk_score(df)

            if rsi_val:
                rsi_color = "#ff4560" if rsi_val > 70 else "#00e5a0" if rsi_val < 30 else "#ffa726"
                if rsi_val > 70:
                    rsi_zone   = "Overbought 🔴"
                    rsi_mean   = "Stock bahut upar aa gaya hai — correction ya pullback ho sakta hai"
                    rsi_action = "Caution: sell ya wait karo"
                elif rsi_val < 30:
                    rsi_zone   = "Oversold 🟢"
                    rsi_mean   = "Stock bahut neeche aa gaya hai — bounce ya recovery possible"
                    rsi_action = "Opportunity: buy consider kar sakte ho"
                elif rsi_val > 55:
                    rsi_zone   = "Bullish Zone 🟡"
                    rsi_mean   = "Momentum positive hai — buyers control mein hain"
                    rsi_action = "Trend follow karo"
                elif rsi_val < 45:
                    rsi_zone   = "Bearish Zone 🟡"
                    rsi_mean   = "Momentum negative hai — sellers control mein hain"
                    rsi_action = "Wait karo ya short"
                else:
                    rsi_zone   = "Neutral ⚪"
                    rsi_mean   = "Koi clear signal nahi — market indecisive hai"
                    rsi_action = "Dusre indicators dekho"

                st.markdown(f"""
                <div class="card">
                    <div class="card-title">RSI (14) — Relative Strength Index</div>
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                        <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:{rsi_color}">{rsi_val:.1f}</div>
                        <div>
                            <div style="font-size:13px;font-weight:600;color:{rsi_color}">{rsi_zone}</div>
                            <div style="font-size:11px;color:#5a6880;margin-top:2px">{rsi_action}</div>
                        </div>
                    </div>
                    <div style="background:#111820;border-radius:6px;height:8px;margin-bottom:10px;position:relative;">
                        <div style="position:absolute;left:0;top:0;height:100%;width:30%;background:#00e5a040;border-radius:6px 0 0 6px;"></div>
                        <div style="position:absolute;right:0;top:0;height:100%;width:30%;background:#ff456040;border-radius:0 6px 6px 0;"></div>
                        <div style="position:absolute;top:-4px;left:calc({min(rsi_val,99):.0f}% - 6px);width:12px;height:16px;background:{rsi_color};border-radius:3px;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-family:'Space Mono',monospace;font-size:9px;color:#5a6880;margin-bottom:10px;">
                        <span>0 — Oversold</span><span>50 — Neutral</span><span>100 — Overbought</span>
                    </div>
                    <div style="font-size:12px;color:#9aa5b8;border-top:1px solid rgba(255,255,255,0.05);padding-top:10px;">
                        💡 {rsi_mean}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── AI SIGNAL — use cache on refresh ──
            if is_refresh and 'pred' in cache:
                pred          = cache['pred']
                current_price = cache.get('current_price', float(df['Close'].iloc[-1]))
            else:
                X, y = prepare_data(df)
                m, acc = train_model(X, y)
                pred          = predict(m, df)
                current_price = float(df['Close'].iloc[-1])
                if stock in st.session_state.analysis_cache:
                    st.session_state.analysis_cache[stock]['pred']          = pred
                    st.session_state.analysis_cache[stock]['current_price'] = current_price

            if pred is not None:
                is_buy    = pred > current_price
                diff      = pred - current_price
                diff_pct  = (diff / current_price) * 100
                signal_label = "BUY 📈" if is_buy else "SELL 📉"
                sig_color = "#00e5a0" if is_buy else "#ff4560"
                sig_bg    = "rgba(0,229,160,0.08)" if is_buy else "rgba(255,69,96,0.08)"
                sig_bdr   = "rgba(0,229,160,0.25)" if is_buy else "rgba(255,69,96,0.25)"

                # Price range: ±1 ATR
                try:
                    import ta as _ta
                    atr = _ta.volatility.AverageTrueRange(
                        df['High'].squeeze(), df['Low'].squeeze(),
                        df['Close'].squeeze(), window=14
                    ).average_true_range().iloc[-1]
                    low_target  = round(pred - atr, 2)
                    high_target = round(pred + atr, 2)
                    has_range   = True
                except Exception:
                    has_range = False

                # Why signal — logic explain
                reasons = []
                if rsi_val:
                    if rsi_val < 35:  reasons.append(f"RSI {rsi_val:.0f} — oversold, bounce likely")
                    elif rsi_val > 65: reasons.append(f"RSI {rsi_val:.0f} — overbought, pullback risk")
                    else:              reasons.append(f"RSI {rsi_val:.0f} — {'bullish' if rsi_val>50 else 'bearish'} momentum")
                if 'ma20' in df.columns and 'ma50' in df.columns:
                    ma20 = float(df['ma20'].iloc[-1])
                    ma50 = float(df['ma50'].iloc[-1])
                    if ma20 > ma50: reasons.append("MA20 > MA50 — uptrend confirmed")
                    else:           reasons.append("MA20 < MA50 — downtrend in play")
                if diff_pct > 0:    reasons.append(f"Model projects +{diff_pct:.2f}% upside")
                else:               reasons.append(f"Model projects {diff_pct:.2f}% downside")

                                # Build HTML strings separately — avoid nested f-string issues
                if has_range:
                    range_section = (
                        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;">'
                        '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:12px 16px;text-align:center;">'
                        '<div style="font-size:10px;color:#5a6880;text-transform:uppercase;letter-spacing:1px;">Conservative</div>'
                        f'<div style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;color:#00e5a0;margin-top:4px;">&#8377;{low_target:,.2f}</div>'
                        '</div>'
                        '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:12px 16px;text-align:center;">'
                        '<div style="font-size:10px;color:#5a6880;text-transform:uppercase;letter-spacing:1px;">Optimistic</div>'
                        f'<div style="font-family:Space Mono,monospace;font-size:16px;font-weight:700;color:#ffa726;margin-top:4px;">&#8377;{high_target:,.2f}</div>'
                        '</div>'
                        '</div>'
                    )
                else:
                    range_section = ""

                reasons_section = ""
                for r in reasons:
                    reasons_section += (
                        '<div style="display:flex;gap:8px;align-items:flex-start;margin-top:8px;">'
                        f'<span style="color:{sig_color};font-size:11px;margin-top:1px;flex-shrink:0;">&#9658;</span>'
                        f'<span style="font-size:13px;color:#9aa5b8;">{r}</span>'
                        '</div>'
                    )

                signal_html = (
                    '<div class="card">'
                    '<div class="card-title">AI Signal &middot; Ensemble Model</div>'
                    f'<div style="background:{sig_bg};border:1px solid {sig_bdr};border-radius:10px;padding:16px 20px;">'
                    '<div style="display:flex;align-items:center;justify-content:space-between;">'
                    f'<div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:{sig_color};">{signal_label}</div>'
                    '<div style="text-align:right;">'
                    f'<div style="font-family:Space Mono,monospace;font-size:18px;font-weight:700;color:{sig_color};">&#8377;{pred:,.2f}</div>'
                    f'<div style="font-family:Space Mono,monospace;font-size:12px;color:#5a6880;">{diff_pct:+.2f}%</div>'
                    '</div>'
                    '</div>'
                    + range_section +
                    '</div>'
                    '<div style="margin-top:14px;border-top:1px solid rgba(255,255,255,0.05);padding-top:12px;">'
                    '<div style="font-family:Space Mono,monospace;font-size:10px;color:#5a6880;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Why this signal?</div>'
                    + reasons_section +
                    '</div>'
                    '</div>'
                )
                st.markdown(signal_html, unsafe_allow_html=True)

            # ── RISK CARD ──
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Risk Level</div>
                <div class="stat-mini">
                    <div class="stat-mini-val" style="font-size:18px;">{risk}</div>
                    <div style="font-size:11px;color:#5a6880;margin-top:4px;">
                        {'Volatility kam hai — price stable rahega' if 'Low' in risk else
                         'Moderate volatility — normal market conditions' if 'Medium' in risk else
                         'High volatility — bade moves possible, careful raho'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 52W HIGH/LOW ──
            try:
                import yfinance as _yf
                _ticker = _yf.Ticker(stock)
                _info   = _ticker.fast_info
                _52h = getattr(_info, 'year_high', None)
                _52l = getattr(_info, 'year_low',  None)
                _cp  = float(df['Close'].squeeze().iloc[-1])

                if _52h and _52l:
                    _pct_from_high = ((_cp - _52h) / _52h) * 100
                    _pct_from_low  = ((_cp - _52l) / _52l) * 100
                    _range_pct     = ((_cp - _52l) / (_52h - _52l)) * 100 if _52h != _52l else 50

                    st.markdown(
                        '<div class="card">'
                        '<div class="card-title">52 Week Range</div>'
                        f'<div style="display:flex;justify-content:space-between;font-family:Space Mono,monospace;font-size:11px;margin-bottom:8px;">'
                        f'<span style="color:#ff4560">L &#8377;{_52l:,.2f}</span>'
                        f'<span style="color:#5a6880">Current &#8377;{_cp:,.2f}</span>'
                        f'<span style="color:#00e5a0">H &#8377;{_52h:,.2f}</span>'
                        f'</span></div>'
                        f'<div style="background:#111820;border-radius:4px;height:8px;margin:8px 0;position:relative;">'
                        f'<div style="position:absolute;left:0;top:0;height:100%;width:{min(_range_pct,100):.1f}%;'
                        f'background:linear-gradient(90deg,#ff4560,#ffa726,#00e5a0);border-radius:4px;"></div>'
                        f'<div style="position:absolute;top:-4px;left:calc({min(_range_pct,100):.1f}% - 6px);'
                        f'width:12px;height:16px;background:#e8edf5;border-radius:3px;"></div>'
                        f'</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;">'
                        f'<div class="stat-mini"><div class="stat-mini-label">From 52W High</div>'
                        f'<div class="stat-mini-val" style="color:#ff4560">{_pct_from_high:.1f}%</div></div>'
                        f'<div class="stat-mini"><div class="stat-mini-label">From 52W Low</div>'
                        f'<div class="stat-mini-val" style="color:#00e5a0">+{_pct_from_low:.1f}%</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
            except Exception as _e:
                print(f"[52W] {_e}")

        # ── TABS FOR DETAILS ──
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧠 Deep Learning", "📰 News", "📉 Backtest", "💹 Paper Trade", "📊 Compare"])

        with tab1:
            def show_dl_result(lstm_price, curr, cached=False):
                """Deep Learning result display with full explanation"""
                import ta as _ta

                is_up    = lstm_price > curr
                diff     = lstm_price - curr
                diff_pct = (diff / curr * 100) if curr else 0
                sig_color = "#00e5a0" if is_up else "#ff4560"
                direction = "📈 Bullish" if is_up else "📉 Bearish"

                # Price range using 3% band
                band = curr * 0.03
                conservative = round(lstm_price - band/2, 2)
                optimistic   = round(lstm_price + band/2, 2)

                # Overall outlook summary
                if diff_pct > 2:
                    outlook = "Strong Bullish 🟢"
                    summary = f"Model strong upside project kar raha hai. Price ₹{conservative:,.0f} se ₹{optimistic:,.0f} tak ja sakta hai."
                elif diff_pct > 0.5:
                    outlook = "Mildly Bullish 🟡"
                    summary = f"Thodi upward movement expected. ₹{conservative:,.0f}–₹{optimistic:,.0f} range likely."
                elif diff_pct > -0.5:
                    outlook = "Sideways ⚪"
                    summary = f"Market flat rahega. ₹{conservative:,.0f}–₹{optimistic:,.0f} ke beech consolidation."
                elif diff_pct > -2:
                    outlook = "Mildly Bearish 🟡"
                    summary = f"Thodi downward pressure. ₹{conservative:,.0f}–₹{optimistic:,.0f} range mein girawat ho sakti hai."
                else:
                    outlook = "Strong Bearish 🔴"
                    summary = f"Model significant downside project kar raha hai. ₹{conservative:,.0f} tak gir sakta hai."

                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                            border-radius:12px;padding:20px 24px;margin-bottom:16px;">
                    <div style="font-family:'Space Mono',monospace;font-size:10px;color:#5a6880;
                                text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;">
                        Deep Learning Prediction · 6 Month Daily Data
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px;">
                        <div style="text-align:center;">
                            <div style="font-size:11px;color:#5a6880;margin-bottom:4px;">Current Price</div>
                            <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;">₹{curr:,.2f}</div>
                        </div>
                        <div style="text-align:center;border-left:1px solid rgba(255,255,255,0.07);
                                    border-right:1px solid rgba(255,255,255,0.07);">
                            <div style="font-size:11px;color:#5a6880;margin-bottom:4px;">Price Target</div>
                            <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:{sig_color}">₹{lstm_price:,.2f}</div>
                            <div style="font-family:'Space Mono',monospace;font-size:11px;color:{sig_color}">{diff_pct:+.2f}%</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:11px;color:#5a6880;margin-bottom:4px;">Direction</div>
                            <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:{sig_color}">{direction}</div>
                        </div>
                    </div>
                    <div style="background:#111820;border-radius:8px;padding:14px 18px;margin-bottom:14px;">
                        <div style="font-size:12px;font-weight:600;color:{sig_color};margin-bottom:6px;">{outlook}</div>
                        <div style="font-size:13px;color:#9aa5b8;line-height:1.6;">{summary}</div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
                        <div class="stat-mini">
                            <div class="stat-mini-label">Conservative Target</div>
                            <div class="stat-mini-val" style="color:#00e5a0">₹{conservative:,.2f}</div>
                        </div>
                        <div class="stat-mini">
                            <div class="stat-mini-label">Optimistic Target</div>
                            <div class="stat-mini-val" style="color:#ffa726">₹{optimistic:,.2f}</div>
                        </div>
                    </div>
                    <div style="border-top:1px solid rgba(255,255,255,0.05);padding-top:12px;">
                        <div style="font-family:'Space Mono',monospace;font-size:10px;color:#5a6880;
                                    text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">How this works</div>
                        <div style="font-size:12px;color:#5a6880;line-height:1.7;">
                            Model ne last 6 months ki daily closing prices, RSI, Bollinger Bands, aur MACD dekhe hain.
                            In patterns ke basis pe agle trading session ka price range predict kiya gaya hai.
                            Conservative = safe estimate · Optimistic = best case scenario.
                        </div>
                    </div>
                    {'<div style="margin-top:10px;font-family:Space Mono,monospace;font-size:10px;color:#5a6880;">📦 Cached result — Re-Analyze button dabao to retrain</div>' if cached else ''}
                </div>
                """, unsafe_allow_html=True)

            # Use cached LSTM result on refresh
            if is_refresh and 'lstm_price' in cache:
                show_dl_result(cache['lstm_price'], cache.get('lstm_curr', cache['lstm_price']), cached=True)
            else:
                df_long = get_long_data(stock)
                if df_long is None or len(df_long) < 60:
                    st.warning("Need 60+ days data for deep learning prediction")
                else:
                    with st.spinner("Deep Learning model training on 6 months data..."):
                        Xl, yl, scaler = prepare_lstm_data(df_long)
                        lstm_m         = train_lstm(Xl, yl)
                        lstm_price     = predict_lstm(lstm_m, df_long, scaler)

                    if lstm_price:
                        curr = float(df_long['Close'].squeeze().iloc[-1])
                        show_dl_result(lstm_price, curr)
                        if stock in st.session_state.analysis_cache:
                            st.session_state.analysis_cache[stock]['lstm_price'] = lstm_price
                            st.session_state.analysis_cache[stock]['lstm_curr']  = curr

        with tab2:
            # Use cached news on refresh
            if is_refresh and 'news_res' in cache:
                res = cache['news_res']
            else:
                with st.spinner("Fetching news..."):
                    news = fetch_news(stock)
                res = analyze_news(news)
                if stock in st.session_state.analysis_cache:
                    st.session_state.analysis_cache[stock]['news_res'] = res

            news_html = '<div class="card"><div class="card-title">Latest News Sentiment</div>'
            for item in res:
                if item["sentiment"] == "POSITIVE":
                    dot = "news-dot-pos"
                    sent_cls = "news-sentiment-pos"
                elif item["sentiment"] == "NEGATIVE":
                    dot = "news-dot-neg"
                    sent_cls = "news-sentiment-neg"
                else:
                    dot = "news-dot-neu"
                    sent_cls = "news-sentiment-neu"

                news_html += f"""
                <div class="news-row">
                    <div class="{dot}"></div>
                    <div>
                        <div class="{sent_cls}">{item['sentiment']} · {item['score']}</div>
                        <div class="news-headline">{item['news'][:120]}{'...' if len(item['news']) > 120 else ''}</div>
                    </div>
                </div>"""

            overall = overall_sentiment(res)
            sentiment_color = "#00e5a0" if "Bullish" in overall else "#ff4560" if "Bearish" in overall else "#5a6880"
            news_html += f"""
            <div style="margin-top:16px;padding:12px 16px;background:rgba(255,255,255,0.03);
                        border-radius:8px;border:1px solid rgba(255,255,255,0.07);">
                <span style="font-family:'Space Mono',monospace;font-size:12px;color:{sentiment_color};font-weight:700;">
                    Overall: {overall}
                </span>
            </div></div>"""
            st.markdown(news_html, unsafe_allow_html=True)

        with tab3:
            # RSI kya hota hai — explanation
            st.markdown("""
            <div style="background:#111820;border:1px solid rgba(255,255,255,0.07);
                        border-radius:10px;padding:16px 20px;margin-bottom:16px;">
                <div style="font-family:'Space Mono',monospace;font-size:10px;color:#00e5a0;
                            text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">
                    RSI kya hota hai?
                </div>
                <div style="font-size:13px;color:#9aa5b8;line-height:1.8;">
                    <b style="color:#e8edf5;">RSI (Relative Strength Index)</b> ek momentum indicator hai jo 0–100 ke beech hota hai.<br><br>
                    <span style="color:#00e5a0;">● RSI &lt; 30</span> — Stock <b style="color:#e8edf5;">Oversold</b> hai → yaani bahut zyada gir gaya hai → 
                    <b style="color:#00e5a0;">Buy ka mauka ho sakta hai</b><br>
                    <span style="color:#ffa726;">● RSI 30–70</span> — <b style="color:#e8edf5;">Normal zone</b> → koi extreme signal nahi<br>
                    <span style="color:#ff4560;">● RSI &gt; 70</span> — Stock <b style="color:#e8edf5;">Overbought</b> hai → bahut upar aa gaya hai → 
                    <b style="color:#ff4560;">Sell/Caution</b><br><br>
                    <b style="color:#e8edf5;">RSI Backtest kya karta hai?</b><br>
                    Yeh strategy test karta hai: <i>Agar RSI &lt; 30 pe buy karo aur RSI &gt; 70 pe sell karo</i> toh 
                    pichle data mein kitna profit/loss hota. ₹10,000 se shuru karke dekhte hain.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if 'rsi' not in df.columns:
                st.warning("RSI data needed for backtest — try 5 Days or 1 Month timeframe")
            else:
                # Init backtest cache
                bt_key = f"bt_result_{stock}"
                if bt_key not in st.session_state:
                    st.session_state[bt_key] = None

                if st.button(f"▶ Run RSI Backtest on {stock}", key=f"bt_{stock}"):
                    with st.spinner("Running backtest..."):
                        result = run_backtest(df)
                    st.session_state[bt_key] = result

                # Show result — from session state (survives rerun)
                result = st.session_state.get(bt_key)
                if result:
                    profit   = result['Profit']
                    final    = result['Final']
                    profit_pct = (profit / 10000) * 100
                    is_profit  = profit >= 0
                    p_color    = "#00e5a0" if is_profit else "#ff4560"
                    p_icon     = "▲" if is_profit else "▼"

                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                                border-radius:12px;padding:20px 24px;margin-top:8px;">
                        <div style="font-family:'Space Mono',monospace;font-size:10px;color:#5a6880;
                                    text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">
                            Backtest Result · RSI Strategy · Starting Capital ₹10,000
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;">
                            <div class="stat-mini">
                                <div class="stat-mini-label">Starting Amount</div>
                                <div class="stat-mini-val">₹10,000</div>
                            </div>
                            <div class="stat-mini">
                                <div class="stat-mini-label">Final Amount</div>
                                <div class="stat-mini-val">₹{final:,.2f}</div>
                            </div>
                            <div class="stat-mini">
                                <div class="stat-mini-label">Profit / Loss</div>
                                <div class="stat-mini-val" style="color:{p_color}">
                                    {p_icon} ₹{abs(profit):,.2f}
                                </div>
                            </div>
                        </div>
                        <div style="background:#111820;border-radius:8px;padding:14px 18px;">
                            <div style="font-size:13px;color:{p_color};font-weight:600;margin-bottom:6px;">
                                {p_icon} {abs(profit_pct):.1f}% {'Profit' if is_profit else 'Loss'} on ₹10,000 investment
                            </div>
                            <div style="font-size:12px;color:#5a6880;line-height:1.7;">
                                {'✅ RSI strategy ne is stock pe achha kaam kiya. Oversold pe buy aur overbought pe sell karne se ' + f'₹{abs(profit):,.0f} ka faida hua.' if is_profit else
                                 '⚠ RSI strategy is stock pe effective nahi rahi. ' + f'₹{abs(profit):,.0f} ka nuksan hua. Alag strategy try karo.'}
                            </div>
                        </div>
                        <div style="margin-top:12px;font-size:11px;color:#5a6880;">
                            ⚠ Disclaimer: Yeh backtest sirf historical data pe hai. Future returns ki guarantee nahi.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab5:
            # ── STOCK COMPARISON ──
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:11px;color:#5a6880;margin-bottom:12px;">
                Compare this stock with another — normalized chart (both start at 100%)
            </div>
            """, unsafe_allow_html=True)

            cmp_col1, cmp_col2 = st.columns([3, 1])
            with cmp_col1:
                compare_input = st.text_input(
                    "Compare with",
                    value=st.session_state.compare_stock,
                    placeholder="e.g. TCS.NS, NIFTY50.NS",
                    label_visibility="collapsed",
                    key=f"cmp_{stock}"
                )
            with cmp_col2:
                cmp_clicked = st.button("Compare →", key=f"cmp_btn_{stock}")

            if cmp_clicked and compare_input.strip():
                cmp_sym = compare_input.strip().upper()
                st.session_state.compare_stock = cmp_sym

                with st.spinner(f"Fetching {cmp_sym}..."):
                    df_cmp, _ = get_live_data(cmp_sym, period=period, interval=interval)

                if df_cmp is not None and len(df_cmp) > 5:
                    # Normalize both to 100 at start
                    import pytz as _pytz
                    try:
                        _ist = _pytz.timezone('Asia/Kolkata')
                        if df_cmp.index.tzinfo is None:
                            df_cmp.index = df_cmp.index.tz_localize('UTC').tz_convert(_ist)
                        else:
                            df_cmp.index = df_cmp.index.tz_convert(_ist)
                        df_cmp.index = df_cmp.index.tz_localize(None)
                    except Exception:
                        pass

                    _s1 = df['Close'].squeeze()
                    _s2 = df_cmp['Close'].squeeze()

                    _n1 = (_s1 / _s1.iloc[0]) * 100
                    _n2 = (_s2 / _s2.iloc[0]) * 100

                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Scatter(
                        x=df.index, y=_n1,
                        mode='lines', name=stock,
                        line=dict(color='#00e5a0', width=2)
                    ))
                    fig_cmp.add_trace(go.Scatter(
                        x=df_cmp.index, y=_n2,
                        mode='lines', name=cmp_sym,
                        line=dict(color='#ffa726', width=2)
                    ))
                    fig_cmp.add_hline(y=100, line_dash="dot",
                                      line_color="#5a6880", line_width=1)

                    fig_cmp.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='#0d1117',
                        plot_bgcolor='#0d1117',
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0),
                        hovermode='x unified',
                        yaxis_title="Normalized Price (Base=100)",
                        font=dict(family='DM Sans', color='#9aa5b8'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                   bgcolor='rgba(0,0,0,0)')
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    # Performance comparison
                    _ret1 = (float(_s1.iloc[-1]) - float(_s1.iloc[0])) / float(_s1.iloc[0]) * 100
                    _ret2 = (float(_s2.iloc[-1]) - float(_s2.iloc[0])) / float(_s2.iloc[0]) * 100
                    _winner = stock if _ret1 > _ret2 else cmp_sym

                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric(f"{stock} Return", f"{_ret1:+.2f}%")
                    c_b.metric(f"{cmp_sym} Return", f"{_ret2:+.2f}%")
                    c_c.metric("Winner 🏆", _winner.replace(".NS",""))
                else:
                    st.error(f"Could not fetch data for {cmp_sym}")
            elif st.session_state.compare_stock:
                st.info(f"Press 'Compare →' to compare {stock} with {st.session_state.compare_stock}")
            else:
                st.info("Stock symbol daalo aur Compare dabao — dono ka normalized performance ek chart pe dikhega")

        with tab4:
            if mode == "Paper Trading":
                price = float(df['Close'].iloc[-1])
                bal = st.session_state.paper["balance"]
                holdings = st.session_state.paper["shares"]

                st.markdown(f"""
                <div style="display:flex;gap:16px;margin-bottom:16px;">
                    <div class="stat-mini" style="flex:1">
                        <div class="stat-mini-label">Balance</div>
                        <div class="stat-mini-val" style="color:#00e5a0">₹{bal:,.2f}</div>
                    </div>
                    <div class="stat-mini" style="flex:1">
                        <div class="stat-mini-label">{stock} Holdings</div>
                        <div class="stat-mini-val">{holdings.get(stock, 0)} shares</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"🟢 BUY @ ₹{price:,.2f}", key=f"buy_{stock}"):
                        st.session_state.paper = buy(st.session_state.paper, stock, price)
                        st.success(f"✅ Bought {stock} @ ₹{price:,.2f}")
                        st.rerun()
                with c2:
                    if st.button(f"🔴 SELL @ ₹{price:,.2f}", key=f"sell_{stock}"):
                        st.session_state.paper = sell(st.session_state.paper, stock, price)
                        st.success(f"✅ Sold {stock} @ ₹{price:,.2f}")
                        st.rerun()
            else:
                st.info("Switch mode to 'Paper Trading' to practice trades")

        st.markdown("<hr>", unsafe_allow_html=True)

# ── AUTO REFRESH (sirf price + chart, baaki cached) ──
if auto_refresh and st.session_state.get("analyzed_stocks"):
    if is_market_open():
        countdown_placeholder = st.empty()
        for i in range(refresh_interval, 0, -1):
            countdown_placeholder.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:11px;color:#5a6880;
                        text-align:center;padding:8px;background:#0d1117;border-radius:6px;
                        border:1px solid rgba(255,255,255,0.05);">
                🔄 Price refresh in {i}s &nbsp;·&nbsp; ML/News cached
            </div>""", unsafe_allow_html=True)
            _time.sleep(1)
        countdown_placeholder.empty()
        # ✅ KEY FIX: set flag so only price+chart refreshes
        st.session_state.price_refresh_only = True
        st.rerun()
    else:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:#5a6880;
                    text-align:center;padding:8px;">
            🔴 Auto refresh paused — Market closed (9:15 AM – 3:30 PM IST)
        </div>""", unsafe_allow_html=True)

# Reset refresh flag after use
if st.session_state.get("price_refresh_only"):
    st.session_state.price_refresh_only = False