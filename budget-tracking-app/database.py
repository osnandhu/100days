from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager
from config import Config

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=Config.SQLALCHEMY_ECHO)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def init_db():
    from models import User, Budget, Transaction, CategoryPattern
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
