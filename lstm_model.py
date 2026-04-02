import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import ta


def prepare_lstm_data(df, lookback=20):
    """
    Sequence-based features with technical indicators.
    lookback=20 — last 20 candles ka full context.
    """
    data = df.copy()
    close = data['Close'].squeeze()

    # Add indicators
    data['rsi']      = ta.momentum.RSIIndicator(close, window=14).rsi()
    data['ma10']     = close.rolling(10).mean()
    data['ma20']     = close.rolling(20).mean()
    data['return_1'] = close.pct_change(1)
    data['return_5'] = close.pct_change(5)

    bb = ta.volatility.BollingerBands(close, window=20)
    data['bb_pos'] = (close - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband() + 1e-8
    )

    macd = ta.trend.MACD(close)
    data['macd_diff'] = macd.macd_diff()

    # Features to use in sequence
    feat_cols = ['return_1', 'return_5', 'rsi', 'bb_pos', 'macd_diff']
    feat_cols = [c for c in feat_cols if c in data.columns]

    # Scale Close separately
    price_scaler = MinMaxScaler()
    prices = price_scaler.fit_transform(data[['Close']].values)

    # Scale features
    feat_scaler = MinMaxScaler()
    feats = data[feat_cols].values
    feats_scaled = feat_scaler.fit_transform(
        pd.DataFrame(feats, columns=feat_cols).fillna(0).values
    )

    # Combine: price + features
    combined = np.hstack([prices, feats_scaled])
    combined = np.nan_to_num(combined, nan=0.0)

    X, y = [], []
    for i in range(lookback, len(combined)):
        X.append(combined[i - lookback:i].flatten())
        y.append(prices[i][0])  # next close (scaled)

    return np.array(X), np.array(y), price_scaler


def train_lstm(X, y):
    """
    Two-model ensemble: GBR + RF on sequence features.
    """
    if len(X) < 30:
        return None

    # GBR — good at capturing non-linear patterns
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )

    # RF — good at reducing variance
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]

    gbr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return (gbr, rf)


def predict_lstm(model_tuple, df, scaler, lookback=20):
    """Predict next closing price"""
    if model_tuple is None:
        return None

    try:
        gbr, rf = model_tuple
        data = df.copy()
        close = data['Close'].squeeze()

        data['rsi']      = ta.momentum.RSIIndicator(close, window=14).rsi()
        data['ma10']     = close.rolling(10).mean()
        data['ma20']     = close.rolling(20).mean()
        data['return_1'] = close.pct_change(1)
        data['return_5'] = close.pct_change(5)

        bb = ta.volatility.BollingerBands(close, window=20)
        data['bb_pos'] = (close - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband() + 1e-8
        )

        macd = ta.trend.MACD(close)
        data['macd_diff'] = macd.macd_diff()

        feat_cols = ['return_1', 'return_5', 'rsi', 'bb_pos', 'macd_diff']
        feat_cols = [c for c in feat_cols if c in data.columns]

        prices_scaled = scaler.transform(data[['Close']].values)

        feat_scaler = MinMaxScaler()
        feats_scaled = feat_scaler.fit_transform(
            data[feat_cols].fillna(0).values
        )

        combined = np.hstack([prices_scaled, feats_scaled])
        combined = np.nan_to_num(combined, nan=0.0)

        if len(combined) < lookback:
            return None

        X_test = combined[-lookback:].flatten().reshape(1, -1)

        gbr_pred = gbr.predict(X_test)[0]
        rf_pred  = rf.predict(X_test)[0]
        avg_pred = (gbr_pred * 0.6 + rf_pred * 0.4)  # GBR ko thoda zyada weight

        price = scaler.inverse_transform([[avg_pred]])[0][0]
        return float(round(price, 2))

    except Exception as e:
        print(f"[LSTM Model] Error: {e}")
        return None
