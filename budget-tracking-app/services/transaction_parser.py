import pandas as pd
from database import get_db_session
from models import Transaction
from services.intelligent_parser import parse_row


def save_parsed_transaction(user_id, description, amount, category, date):
    with get_db_session() as db:
        txn = Transaction(
            user_id=user_id,
            description=description,
            amount=amount,
            category=category,
            date=date,
        )
        db.add(txn)
        return txn


def process_csv_with_intelligence(csv_path, user_id):
    df = pd.read_csv(csv_path)
    auto_saved = []
    needs_review = []

    for _, row in df.iterrows():
        parsed = parse_row(row)
        if parsed["confidence"] == "high":
            save_parsed_transaction(
                user_id=user_id,
                description=parsed["description"],
                amount=parsed["amount"],
                category=parsed["category"],
                date=parsed["date"],
            )
            auto_saved.append(parsed)
        else:
            needs_review.append(parsed)

    return auto_saved, needs_review
