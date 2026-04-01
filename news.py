import requests
from textblob import TextBlob

NEWSAPI_KEY = "c81e872af5f1407ca516f53761eacb2b"  # 👈 apna key yahan daalo


def fetch_news(stock):
    query = stock.replace(".NS", "").replace(".BO", "")

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5,
            "apiKey": NEWSAPI_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            return _fallback_news(query)

        articles = data.get("articles", [])
        if not articles:
            return _fallback_news(query)

        return [a["title"] for a in articles if a.get("title")][:5]

    except Exception as e:
        print(f"News fetch failed: {e}")
        return _fallback_news(query)


def _fallback_news(query):
    return [
        f"{query} stock market performance today",
        f"{query} quarterly results analysis"
    ]


def analyze_news(news_list):
    results = []
    for n in news_list:
        try:
            blob = TextBlob(n)
            polarity = blob.sentiment.polarity  # -1 to +1

            if polarity > 0.05:
                label = "POSITIVE"
                score = round(0.5 + polarity / 2, 2)
            elif polarity < -0.05:
                label = "NEGATIVE"
                score = round(0.5 - polarity / 2, 2)
            else:
                label = "NEUTRAL"
                score = round(0.5, 2)

            results.append({
                "news":      n,
                "sentiment": label,
                "score":     min(score, 0.99)
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
        elif r['sentiment'] == "NEGATIVE":
            score -= r['score']

    if score > 0.3:
        return "Bullish 🟢"
    elif score < -0.3:
        return "Bearish 🔴"
    else:
        return "Neutral ⚪"
