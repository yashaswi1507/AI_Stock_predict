def run_backtest(df):
    balance = 10000
    shares = 0

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        rsi = df['rsi'].iloc[i]

        if rsi < 30 and balance > 0:
            shares = balance / price
            balance = 0

        elif rsi > 70 and shares > 0:
            balance = shares * price
            shares = 0

    final = balance + shares * df['Close'].iloc[-1]

    return {
        "Final": round(final, 2),
        "Profit": round(final - 10000, 2)
    }