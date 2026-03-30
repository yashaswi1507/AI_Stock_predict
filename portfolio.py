def init_portfolio():
    return []

def add_stock(p, s, price, qty):
    p.append({"stock": s, "buy": price, "qty": qty})
    return p

def calculate(p, prices):
    total = 0
    data = []

    for i in p:
        cur = prices.get(i["stock"], i["buy"])
        profit = (cur - i["buy"]) * i["qty"]
        total += cur * i["qty"]

        data.append({
            "Stock": i["stock"],
            "Profit": round(profit, 2)
        })

    return total, data