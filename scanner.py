from data import get_data, get_long_data

def get_top_movers():
    stocks = ["AAPL", "TSLA", "MSFT", "GOOG"]
    res = []

    for s in stocks:
        df = get_data(s)

        if df.empty or len(df) < 2:
            continue

        # percent change (scalar value)
        change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

        res.append((s, float(change)))  # ensure scalar

    # sort properly
    res.sort(key=lambda x: x[1], reverse=True)

    return res