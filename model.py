import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ta


def build_features(df):
    """
    Strong feature engineering — yahi accuracy ka asli raaz hai.
    Sirf RSI/MA se kaam nahi chalta — price action + momentum + volatility sab chahiye.
    """
    data = df.copy()

    close = data['Close'].squeeze()
    high  = data['High'].squeeze()  if 'High'  in data.columns else close
    low   = data['Low'].squeeze()   if 'Low'   in data.columns else close
    vol   = data['Volume'].squeeze() if 'Volume' in data.columns else pd.Series(0, index=data.index)

    # ── Price action features ──
    data['return_1']  = close.pct_change(1)
    data['return_3']  = close.pct_change(3)
    data['return_5']  = close.pct_change(5)
    data['return_10'] = close.pct_change(10)

    # ── Moving averages ──
    data['ma5']  = close.rolling(5).mean()
    data['ma10'] = close.rolling(10).mean()
    data['ma20'] = close.rolling(20).mean()
    data['ma50'] = close.rolling(50).mean()

    # Price vs MA ratio (trend strength)
    data['price_ma5_ratio']  = close / data['ma5']
    data['price_ma20_ratio'] = close / data['ma20']
    data['ma5_ma20_ratio']   = data['ma5'] / data['ma20']

    # ── Momentum indicators ──
    data['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    data['rsi_3'] = ta.momentum.RSIIndicator(close, window=3).rsi()   # short term RSI

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14)
    data['stoch_k'] = stoch.stoch()
    data['stoch_d'] = stoch.stoch_signal()

    # ── Trend indicators ──
    macd = ta.trend.MACD(close)
    data['macd']        = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff']   = macd.macd_diff()

    # EMA
    data['ema9']  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    data['ema21'] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    data['ema_ratio'] = data['ema9'] / data['ema21']

    # ── Volatility indicators ──
    bb = ta.volatility.BollingerBands(close, window=20)
    data['bb_upper']  = bb.bollinger_hband()
    data['bb_lower']  = bb.bollinger_lband()
    data['bb_width']  = (data['bb_upper'] - data['bb_lower']) / data['ma20']
    data['bb_pos']    = (close - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-8)

    data['atr'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    data['atr_pct'] = data['atr'] / close

    # ── Volume features ──
    if vol.sum() > 0:
        data['vol_ma10']   = vol.rolling(10).mean()
        data['vol_ratio']  = vol / (data['vol_ma10'] + 1e-8)
        data['vol_change'] = vol.pct_change()
    else:
        data['vol_ratio']  = 1.0
        data['vol_change'] = 0.0

    # ── Price pattern features ──
    data['high_low_range'] = (high - low) / close       # candle range
    data['close_position'] = (close - low) / (high - low + 1e-8)  # close position in range

    # Lag features (past prices)
    for lag in [1, 2, 3, 5]:
        data[f'lag_{lag}'] = close.shift(lag)
        data[f'lag_{lag}_ret'] = data[f'lag_{lag}'].pct_change()

    # Rolling stats
    data['rolling_std_5']  = close.rolling(5).std() / close
    data['rolling_std_10'] = close.rolling(10).std() / close
    data['rolling_max_10'] = close.rolling(10).max() / close
    data['rolling_min_10'] = close.rolling(10).min() / close

    return data


FEATURE_COLS = [
    'return_1', 'return_3', 'return_5', 'return_10',
    'price_ma5_ratio', 'price_ma20_ratio', 'ma5_ma20_ratio',
    'rsi', 'rsi_3', 'stoch_k', 'stoch_d',
    'macd', 'macd_signal', 'macd_diff',
    'ema_ratio',
    'bb_width', 'bb_pos',
    'atr_pct',
    'vol_ratio', 'vol_change',
    'high_low_range', 'close_position',
    'lag_1_ret', 'lag_2_ret', 'lag_3_ret', 'lag_5_ret',
    'rolling_std_5', 'rolling_std_10',
    'rolling_max_10', 'rolling_min_10',
]


def prepare_data(df):
    if df is None or len(df) < 60:
        return None, None

    data = build_features(df)

    # Target: next candle ka % return (direction + magnitude)
    data['target'] = data['Close'].squeeze().pct_change().shift(-1)

    # Only use available features
    available = [c for c in FEATURE_COLS if c in data.columns]
    data = data[available + ['target', 'Close']].dropna()

    if len(data) < 30:
        return None, None

    X = data[available]
    y = data['target']

    return X, y


def train_model(X, y):
    if X is None or y is None or len(X) < 30:
        return None, 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Ensemble: RF + GBR combined
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    gbr = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )

    rf.fit(X_train, y_train)
    gbr.fit(X_train, y_train)

    rf_score  = rf.score(X_test, y_test)
    gbr_score = gbr.score(X_test, y_test)

    return (rf, gbr), max(rf_score, gbr_score)


def predict(model_tuple, df):
    if model_tuple is None or df is None:
        return None

    try:
        rf, gbr = model_tuple
        data = build_features(df)
        available = [c for c in FEATURE_COLS if c in data.columns]
        data = data[available].dropna()

        if len(data) == 0:
            return None

        latest = data.iloc[-1:][available]

        # Ensemble prediction (average of both)
        rf_pred  = rf.predict(latest)[0]
        gbr_pred = gbr.predict(latest)[0]
        avg_return = (rf_pred + gbr_pred) / 2

        # Convert % return to price
        current_price = float(df['Close'].squeeze().iloc[-1])
        predicted_price = current_price * (1 + avg_return)

        return round(predicted_price, 2)

    except Exception as e:
        print(f"[Predict] Error: {e}")
        return None


def risk_score(df):
    """Better risk score using multiple factors"""
    try:
        close = df['Close'].squeeze()
        vol = close.pct_change().std()

        # ATR based risk
        if 'High' in df.columns and 'Low' in df.columns:
            atr = ta.volatility.AverageTrueRange(
                df['High'].squeeze(), df['Low'].squeeze(), close, window=14
            ).average_true_range().iloc[-1]
            atr_pct = atr / float(close.iloc[-1])
        else:
            atr_pct = vol

        if atr_pct < 0.01:
            return "Low 🟢"
        elif atr_pct < 0.025:
            return "Medium 🟡"
        else:
            return "High 🔴"

    except Exception:
        vol = df['Close'].squeeze().pct_change().std()
        if vol < 0.01:   return "Low 🟢"
        elif vol < 0.02: return "Medium 🟡"
        else:            return "High 🔴"
