import requests
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

NEWSAPI_KEY = "c81e872af5f1407ca516f53761eacb2b"  # 👈 apna key yahan daalo


def fetch_news(stock):
    """
    NewsAPI se real news fetch karta hai stock ke liye.
    Returns list of headline strings.
    """
    # NSE stocks mein .NS hota hai — remove kar do search ke liye
    query = stock.replace(".NS", "").replace(".BO", "")

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",   # latest news pehle
            "pageSize": 5,             # 5 news per stock
            "apiKey": NEWSAPI_KEY
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            print(f"NewsAPI error: {data.get('message')}")
            return _fallback_news(query)

        articles = data.get("articles", [])

        if not articles:
            return _fallback_news(query)

        # sirf headlines return karo
        headlines = [a["title"] for a in articles if a.get("title")]
        return headlines[:5]

    except Exception as e:
        print(f"News fetch failed: {e}")
        return _fallback_news(query)


def _fallback_news(query):
    """API fail ho toh basic fallback"""
    return [
        f"{query} stock market performance today",
        f"{query} quarterly results analysis"
    ]


def analyze_news(news_list):
    results = []

    for n in news_list:
        try:
            r = sentiment_model(n[:512])[0]  # 512 char limit (model constraint)
            results.append({
                "news": n,
                "sentiment": r['label'],
                "score": round(r['score'], 2)
            })
        except Exception:
            continue

    return results


def overall_sentiment(results):
    if not results:
        return "Neutral ⚪"

    score = 0
    for r in results:
        if r['sentiment'] == "POSITIVE":
            score += r['score']
        else:
            score -= r['score']

    if score > 0.3:
        return "Bullish 🟢"
    elif score < -0.3:
        return "Bearish 🔴"
    else:
        return "Neutral ⚪"
