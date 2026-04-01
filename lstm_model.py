# LSTM replaced with GradientBoostingRegressor
# Reason: TensorFlow Python 3.14 support nahi karta (Streamlit Cloud)
# GBR accuracy almost same hai sequence prediction ke liye

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler


def prepare_lstm_data(df, lookback=10):
    """
    Last `lookback` days ka data features mein convert karo.
    Har row = [close_t-n, ..., close_t-1] → target = close_t
    """
    scaler = MinMaxScaler()
    prices = scaler.fit_transform(df[['Close']].values)

    X, y = [], []
    for i in range(lookback, len(prices)):
        X.append(prices[i - lookback:i].flatten())
        y.append(prices[i][0])

    return np.array(X), np.array(y), scaler


def train_lstm(X, y):
    """GradientBoosting model train karo"""
    if len(X) < 20:
        return None

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X, y)
    return model


def predict_lstm(model, df, scaler, lookback=10):
    """Next price predict karo"""
    if model is None:
        return None

    try:
        prices = scaler.transform(df[['Close']].tail(lookback).values)
        X_test = prices.flatten().reshape(1, -1)
        pred_scaled = model.predict(X_test)
        pred_price = scaler.inverse_transform([[pred_scaled[0]]])[0][0]
        return float(pred_price)
    except Exception as e:
        print(f"[GBR Predict] Error: {e}")
        return None
