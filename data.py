import yfinance as yf
import ta
import pandas as pd
import requests
from datetime import datetime, timedelta

# ── Try importing optional libraries ──
try:
    from openchart import NSEData
    _nse = NSEData()
    OPENCHART_OK = True
except Exception:
    OPENCHART_OK = False

try:
    from nsetools import Nse
    _nsetools = Nse()
    NSETOOLS_OK = True
except Exception:
    NSETOOLS_OK = False


# ── Symbol conversion helpers ──
def _to_openchart_symbol(stock):
    """RELIANCE.NS → RELIANCE-EQ"""
    sym = stock.replace(".NS", "").replace(".BO", "")
    return f"{sym}-EQ"


def _to_nsetools_symbol(stock):
    """RELIANCE.NS → reliance (nsetools lowercase)"""
    return stock.replace(".NS", "").replace(".BO", "").lower()


# ── OpenChart: intraday + historical OHLCV ──
def _fetch_openchart(stock, period="1d", interval="5m"):
    """
    NSE se real data via openchart.
    period: 1d, 5d, 1mo, 3mo, 6mo
    interval: 1m, 3m, 5m, 15m, 30m, 1h, 1d
    """
    if not OPENCHART_OK:
        return None

    try:
        sym = _to_openchart_symbol(stock)
        end = datetime.now()

        period_days = {
            "1d": 1, "5d": 5, "1mo": 30,
            "3mo": 90, "6mo": 180
        }
        days = period_days.get(period, 1)
        start = end - timedelta(days=days)

        # interval mapping
        interval_map = {
            "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d",
            "1m": "1m", "3m": "3m", "30m": "30m"
        }
        oc_interval = interval_map.get(interval, "5m")

        df = _nse.historical(sym, 'EQ', start, end, oc_interval)

        if df is None or df.empty:
            return None

        # standardize columns
        df.columns = [c.title() for c in df.columns]
        df.index.name = 'Datetime'

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

        return df if not df.empty else None

    except Exception as e:
        print(f"[OpenChart] Failed: {e}")
        return None


# ── nsetools: live quote (current price only) ──
def get_live_quote(stock):
    """
    Real-time last traded price from NSE.
    Returns dict with lastPrice, change, pChange etc.
    """
    if not NSETOOLS_OK:
        return None

    try:
        sym = _to_nsetools_symbol(stock)
        quote = _nsetools.get_quote(sym)
        return quote
    except Exception as e:
        print(f"[nsetools] Quote failed: {e}")
        return None


# ── yfinance fallback ──
def _fetch_yfinance(stock, period, interval):
    try:
        df = yf.download(stock, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        return clean_df(df)
    except Exception as e:
        print(f"[yfinance] Failed: {e}")
        return None


# ── clean_df: normalize any DataFrame to standard OHLCV ──
def clean_df(df):
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]).strip()
                      for col in df.columns]

    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'open'   in cl and 'Open'   not in rename_map.values():
            rename_map[col] = 'Open'
        elif 'high'  in cl and 'High'   not in rename_map.values():
            rename_map[col] = 'High'
        elif 'low'   in cl and 'Low'    not in rename_map.values():
            rename_map[col] = 'Low'
        elif 'close' in cl and 'Close'  not in rename_map.values():
            rename_map[col] = 'Close'
        elif 'volume' in cl and 'Volume' not in rename_map.values():
            rename_map[col] = 'Volume'

    df.rename(columns=rename_map, inplace=True)

    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in df.columns:
            continue
        val = df[col]
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]
        df[col] = pd.to_numeric(val, errors='coerce')

    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    return df if not df.empty else None


# ── Technical indicators ──
def add_indicators(df):
    if df is None or len(df) < 15:
        return df
    if len(df) > 14:
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    if len(df) > 20:
        df['ma20'] = df['Close'].rolling(20).mean()
    if len(df) > 50:
        df['ma50'] = df['Close'].rolling(50).mean()
    return df


# ── MAIN: get_live_data ──
def get_live_data(stock, period="1d", interval="5m"):
    """
    Priority:
    1. OpenChart (NSE real data, no API key)
    2. yfinance (15min delayed fallback)

    Returns: (df, source)
    source = "nse_live" | "delayed"
    """

    # 1. OpenChart — NSE direct
    print(f"[NSE] Trying OpenChart for {stock}...")
    df = _fetch_openchart(stock, period=period, interval=interval)

    if df is not None:
        print(f"[NSE] ✅ OpenChart success — {len(df)} candles")

        # inject live last price from nsetools if market is open
        quote = get_live_quote(stock)
        if quote and 'lastPrice' in quote:
            live_price = float(str(quote['lastPrice']).replace(',', ''))
            if live_price > 0:
                df.iloc[-1, df.columns.get_loc('Close')] = live_price
                print(f"[NSE] Live price injected: ₹{live_price}")

        df = add_indicators(df)
        return df, "nse_live"

    # 2. yfinance fallback
    print(f"[Fallback] OpenChart failed, using yfinance...")
    df = _fetch_yfinance(stock, period, interval)

    if df is not None:
        df = add_indicators(df)
        return df, "delayed"

    return None, None


# ── scanner.py ke liye ──
def get_data(stock):
    df, _ = get_live_data(stock, period="1d", interval="5m")
    return df


# ── LSTM ke liye 6 month daily data ──
def get_long_data(stock):
    print(f"[LSTM] Fetching long data for {stock}...")

    # OpenChart se 6 month daily
    df = _fetch_openchart(stock, period="6mo", interval="1d")
    if df is not None:
        print(f"[LSTM] ✅ OpenChart — {len(df)} days")
        return df

    # fallback
    df = _fetch_yfinance(stock, "6mo", "1d")
    return df
