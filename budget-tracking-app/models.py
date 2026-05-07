from sqlalchemy import Column, Float, Integer, String, Numeric, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=func.now())

    budget = relationship("Budget", back_populates="user", uselist=False)
    transactions = relationship("Transaction", back_populates="user")
    category_patterns = relationship("CategoryPattern", back_populates="user")


class Budget(Base):
    __tablename__ = "budgets"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    monthly_salary = Column(Numeric(10, 2), nullable=False)
    total_expenditure = Column(Numeric(10, 2), default=0)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="budget")

    @property
    def savings(self):
        return float(self.monthly_salary) - float(self.total_expenditure)

    @property
    def spending_percentage(self):
        if float(self.monthly_salary) == 0:
            return 0
        return (float(self.total_expenditure) / float(self.monthly_salary)) * 100

    @property
    def savings_percentage(self):
        return 100 - self.spending_percentage


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(String(200), nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    category = Column(String(50))
    date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="transactions")


class CategoryPattern(Base):
    __tablename__ = "category_patterns"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    pattern = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    frequency = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="category_patterns")


class InflationForecast(Base):
    """
    Stores inflation forecast results so we can:
    - Track predictions over time (were we accurate?)
    - Show historical forecasts in the dashboard
    - Avoid re-running expensive Prophet fits for the same data
    """
    __tablename__ = "inflation_forecasts"

    id = Column(Integer, primary_key=True)
    forecast_date = Column(String(7), nullable=False)  # "2025-07"
    predicted_inflation = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=False)
    upper_bound = Column(Float, nullable=False)
    confidence_level = Column(String(50), default="95%")
    data_source = Column(String(100))
    model_used = Column(String(50), default="prophet")
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Forecast {self.forecast_date}: {self.predicted_inflation:.2f}%>"


class SpendingAnomaly(Base):
    """
    Persists detected anomalies so we can:
    - Track whether flagged anomalies were real (user feedback)
    - Build a history of anomaly patterns
    - Show anomaly trends on a dashboard
    """
    __tablename__ = "spending_anomalies"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    severity = Column(String(10), nullable=False)  # LOW/MEDIUM/HIGH/CRITICAL
    detection_method = Column(String(30), nullable=False)
    z_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User")
    transaction = relationship("Transaction")
