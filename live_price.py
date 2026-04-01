import requests
import time
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

def is_market_open():
    """NSE market 9:15 AM - 3:30 PM IST, Mon-Fri"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    return market_open <= now <= market_close


def _get_nse_session():
    """
    NSE ke liye proper 3-step session:
    1. Homepage → main cookies
    2. Market page → extra auth cookies  
    3. Ab API call karo
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        # Step 1: Homepage
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)

        # Step 2: Market data page (zaroori cookies milte hain)
        session.get(
            "https://www.nseindia.com/market-data/live-equity-market",
            timeout=10
        )
        time.sleep(0.5)

    except Exception as e:
        print(f"[NSE Session] Warning: {e}")

    return session


def get_current_price(stock):
    """
    NSE se live price fetch karo.
    Market open  → real-time last traded price
    Market closed → previous close
    """
    sym = stock.replace(".NS", "").replace(".BO", "").upper()

    try:
        session = _get_nse_session()

        url = f"https://www.nseindia.com/api/quote-equity?symbol={sym}"
        resp = session.get(url, timeout=10)

        if resp.status_code == 401:
            print(f"[NSE] 401 — session expired, retrying...")
            time.sleep(2)
            session = _get_nse_session()
            resp = session.get(url, timeout=10)

        if resp.status_code != 200:
            print(f"[NSE] Status {resp.status_code} for {sym}")
            return _yfinance_fallback(stock)

        data = resp.json()
        price_info = data.get("priceInfo", {})

        last_price = float(price_info.get("lastPrice", 0) or 0)
        prev_close = float(price_info.get("previousClose", 0) or 0)
        display    = last_price if last_price > 0 else prev_close

        return {
            "symbol":      sym,
            "price":       display,
            "open":        float(price_info.get("open", 0) or 0),
            "high":        float(price_info.get("intraDayHighLow", {}).get("max", 0) or 0),
            "low":         float(price_info.get("intraDayHighLow", {}).get("min", 0) or 0),
            "close":       prev_close,
            "change":      float(price_info.get("change", 0) or 0),
            "pChange":     float(price_info.get("pChange", 0) or 0),
            "volume":      data.get("marketDeptOrderBook", {}).get(
                               "tradeInfo", {}).get("totalTradedVolume", 0),
            "time":        datetime.now().strftime("%H:%M:%S"),
            "market_open": is_market_open(),
            "source":      "nse_live"
        }

    except Exception as e:
        print(f"[NSE Live] Error for {sym}: {e}")
        return _yfinance_fallback(stock)


def _yfinance_fallback(stock):
    """NSE fail ho toh yfinance se last price lo"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(stock)
        info = ticker.fast_info

        price = getattr(info, 'last_price', None) or \
                getattr(info, 'previous_close', None)

        if not price:
            return None

        prev  = getattr(info, 'previous_close', price)
        change  = price - prev
        pchange = (change / prev * 100) if prev else 0

        return {
            "symbol":      stock.replace(".NS","").upper(),
            "price":       round(float(price), 2),
            "open":        round(float(getattr(info, 'open', 0) or 0), 2),
            "high":        round(float(getattr(info, 'day_high', 0) or 0), 2),
            "low":         round(float(getattr(info, 'day_low', 0) or 0), 2),
            "close":       round(float(prev), 2),
            "change":      round(float(change), 2),
            "pChange":     round(float(pchange), 2),
            "volume":      getattr(info, 'three_month_average_volume', 0),
            "time":        datetime.now().strftime("%H:%M:%S"),
            "market_open": is_market_open(),
            "source":      "yfinance_live"
        }
    except Exception as e:
        print(f"[yfinance fallback] Error: {e}")
        return None
