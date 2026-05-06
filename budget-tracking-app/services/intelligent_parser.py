"""
Intelligent parser for bank statement CSV rows.
Categorises transactions based on known patterns and keywords.
"""

CATEGORY_KEYWORDS = {
    "Groceries": ["ntuc", "fairprice", "cold storage", "sheng siong", "giant"],
    "Transport": ["grab", "gojek", "comfort", "ez-link", "smrt", "bus", "mrt"],
    "Food": ["mcdonald", "kfc", "starbucks", "kopitiam", "hawker", "food"],
    "Utilities": ["sp group", "singtel", "starhub", "m1", "electricity", "water"],
    "Rent": ["rent", "lease", "housing"],
    "Entertainment": ["netflix", "spotify", "cinema", "movie"],
    "Shopping": ["shopee", "lazada", "amazon", "uniqlo", "zara"],
    "Subscriptions": ["subscription", "monthly", "annual plan"],
    "Fuel": ["shell", "esso", "caltex", "spc", "petrol"],
}


def categorise_transaction(description):
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return {"category": category, "confidence": "high", "entity": keyword}
    return {"category": None, "confidence": "low", "entity": None}


def parse_row(row):
    description = str(row.get("description", row.get("Description", "")))
    amount = float(row.get("amount", row.get("Amount", 0)))
    date = row.get("date", row.get("Date", None))

    result = categorise_transaction(description)
    return {
        "description": description,
        "amount": amount,
        "date": date,
        "category": result["category"],
        "confidence": result["confidence"],
        "entity": result["entity"],
    }
