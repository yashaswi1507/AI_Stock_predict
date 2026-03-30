import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[['Close']])

    X, y = [], []

    for i in range(60, len(data)):
        X.append(data[i-60:i])
        y.append(data[i])

    return np.array(X), np.array(y), scaler


def train_lstm(X, y):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(64))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32)

    return model


def predict_lstm(model, df, scaler):
    last_60 = df[['Close']].tail(60)
    scaled = scaler.transform(last_60)

    X_test = np.array([scaled])
    pred = model.predict(X_test)

    return scaler.inverse_transform(pred)[0][0]