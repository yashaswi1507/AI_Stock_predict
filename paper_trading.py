def init_paper():
    return {"balance": 10000, "shares": {}}

def buy(p, s, price):
    qty = int(p["balance"] // price)
    p["shares"][s] = p["shares"].get(s, 0) + qty
    p["balance"] -= qty * price
    return p

def sell(p, s, price):
    if s in p["shares"]:
        qty = p["shares"][s]
        p["balance"] += qty * price
        p["shares"][s] = 0
    return p