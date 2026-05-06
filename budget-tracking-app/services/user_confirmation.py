from database import get_db_session
from models import Transaction, CategoryPattern


def confirm_and_save(user_id, parsed_txn, confirmed_category):
    with get_db_session() as db:
        txn = Transaction(
            user_id=user_id,
            description=parsed_txn["description"],
            amount=parsed_txn["amount"],
            category=confirmed_category,
            date=parsed_txn["date"],
        )
        db.add(txn)

        merchant = parsed_txn.get("entity") or parsed_txn["description"][:50]
        existing = (
            db.query(CategoryPattern)
            .filter_by(user_id=user_id, pattern=merchant)
            .first()
        )
        if not existing:
            pattern = CategoryPattern(
                user_id=user_id, pattern=merchant, category=confirmed_category
            )
            db.add(pattern)

        return txn
