from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import settings

# Connection pool to Postgres. One per process, reused everywhere.
engine = create_engine(settings.database_url)

# Factory for DB sessions. Call SessionLocal() to get one.
# autoflush=False: we commit explicitly instead of on every query.
# expire_on_commit=False: keep using an object right after commit() without a refetch.
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    # FastAPI dependency: gives a route a session, closes it when the request ends.
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
