"""
Shared test fixtures for the multi-agent system.

WHAT ARE FIXTURES?
  pytest fixtures are reusable setup/teardown functions. Instead of
  repeating database setup in every test, define it once here and
  inject it by name. pytest discovers conftest.py automatically.
"""

import sys
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, ".")

from database import Base
from models import Budget, CategoryPattern, Transaction, User


@pytest.fixture
def test_engine():
    """
    Create an in-memory SQLite database for testing.

    WHY IN-MEMORY:
    - Tests run faster (no disk I/O)
    - Each test gets a clean database (no leftover data)
    - No cleanup needed after tests
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def test_session(test_engine):
    """Create a database session for testing."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_user(test_session):
    """Create a test user with a budget."""
    user = User(name="Test User", email="test@budget.app")
    test_session.add(user)
    test_session.flush()

    budget = Budget(user_id=user.id, monthly_salary=5000, total_expenditure=3500)
    test_session.add(budget)
    test_session.commit()

    _ = user.budget
    return user


@pytest.fixture
def sample_transactions(test_session, sample_user):
    """
    Create a set of test transactions with known anomalies.

    We plant specific anomalies so tests can verify detection:
    - Transaction 5 (Groceries $500): should be flagged (normal is ~$80)
    - Transaction 10 (Entertainment $300): should be flagged (normal is ~$30)
    """
    categories = {
        "Groceries": [80, 85, 75, 90, 500, 82, 78, 88],  # 500 is anomaly
        "Transport": [50, 45, 55, 48, 52, 47],
        "Food": [25, 30, 28, 35, 22, 27],
        "Entertainment": [30, 25, 35, 28, 300, 32],  # 300 is anomaly
    }

    transactions = []
    base_date = datetime.now() - timedelta(days=60)

    for category, amounts in categories.items():
        for i, amount in enumerate(amounts):
            txn = Transaction(
                user_id=sample_user.id,
                description=f"Test {category} #{i+1}",
                amount=amount,
                category=category,
                date=base_date + timedelta(days=i * 3),
            )
            test_session.add(txn)
            transactions.append(txn)

    test_session.commit()
    return transactions


@pytest.fixture
def sample_cpi_data():
    """Historical CPI data matching World Bank format."""
    return [
        {"year": 2015, "cpi_inflation": -0.52},
        {"year": 2016, "cpi_inflation": -0.53},
        {"year": 2017, "cpi_inflation": 0.58},
        {"year": 2018, "cpi_inflation": 0.44},
        {"year": 2019, "cpi_inflation": 0.57},
        {"year": 2020, "cpi_inflation": -0.18},
        {"year": 2021, "cpi_inflation": 2.30},
        {"year": 2022, "cpi_inflation": 6.12},
        {"year": 2023, "cpi_inflation": 4.82},
        {"year": 2024, "cpi_inflation": 2.40},
    ]
