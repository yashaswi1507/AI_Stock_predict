import yfinance as yf
import ta
import pandas as pd


def clean_df(df):
    if df is None or df.empty:
        return None

    # FIX 1: flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]

    # FIX 2: rename columns (case-insensitive match)
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower and 'Open' not in rename_map.values():
            rename_map[col] = 'Open'
        elif 'high' in col_lower and 'High' not in rename_map.values():
            rename_map[col] = 'High'
        elif 'low' in col_lower and 'Low' not in rename_map.values():
            rename_map[col] = 'Low'
        elif 'close' in col_lower and 'Close' not in rename_map.values():
            rename_map[col] = 'Close'
        elif 'volume' in col_lower and 'Volume' not in rename_map.values():
            rename_map[col] = 'Volume'

    df.rename(columns=rename_map, inplace=True)

    # FIX 3: safely convert each OHLC column to 1D numeric Series
    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in df.columns:
            continue

        val = df[col]

        # ✅ KEY FIX: if it's a DataFrame (2D), take first column
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]

        # ✅ Convert to numeric, drop bad values
        df[col] = pd.to_numeric(val, errors='coerce')

    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

    return df if not df.empty else None


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


def get_data(stock):
    df = yf.download(stock, period="5d", interval="5m", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return None

    df = clean_df(df)
    if df is None:
        return None

    df = add_indicators(df)
    return df


def get_long_data(stock):
    df = yf.download(stock, period="6mo", interval="1d", auto_adjust=True, progress=False)

    if df is None or df.empty:
        return None

    df = clean_df(df)
    return df
