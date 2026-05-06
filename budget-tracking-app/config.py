import os


class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///budget.db")
    SQLALCHEMY_ECHO = os.getenv("SQL_ECHO", "false").lower() == "true"
