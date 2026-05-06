from database import get_db_session
from models import User, Budget


def create_user(name, location, monthly_salary, total_expenditure, intent):
    with get_db_session() as db:
        email = f"{name.lower().replace(' ', '.')}@budget.app"
        user = User(name=name, email=email)
        db.add(user)
        db.flush()

        budget = Budget(
            user_id=user.id,
            monthly_salary=monthly_salary,
            total_expenditure=total_expenditure,
        )
        db.add(budget)
        db.flush()

        _ = user.budget
        return user


def get_all_users():
    with get_db_session() as db:
        users = db.query(User).all()
        for user in users:
            _ = user.budget
        return users
