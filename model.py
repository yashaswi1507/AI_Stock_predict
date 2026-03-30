from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def prepare_data(df):
    required = ['rsi', 'ma20', 'ma50']

    # check columns
    for col in required:
        if col not in df.columns:
            return None, None

    # features
    X = df[required]

    # 🎯 target = NEXT PRICE (continuous)
    y = df['Close'].shift(-1)

    # combine (alignment fix)
    data = X.copy()
    data['target'] = y

    # remove NaN
    data.dropna(inplace=True)

    X = data[required]
    y = data['target']

    return X, y


def train_model(X, y):

    if X is None or y is None or len(X) < 20:
        return None, 0

    if len(X) != len(y):
        return None, 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 🔥 REGRESSOR (NOT CLASSIFIER)
    model = RandomForestRegressor(n_estimators=100)

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)  # R² score

    return model, score


def predict(model, df):
    if model is None or df is None or len(df) == 0:
        return None
    
    df = df.dropna()

    if len(df) == 0:
        return None

    latest = df[['rsi', 'ma20', 'ma50']].iloc[-1:]

    pred = model.predict(latest)

    # 🔥 ensure scalar value
    return float(pred[0])


def risk_score(df):
    vol = df['Close'].pct_change().std()

    if vol < 0.01:
        return "Low 🟢"
    elif vol < 0.02:
        return "Medium 🟡"
    else:
        return "High 🔴"