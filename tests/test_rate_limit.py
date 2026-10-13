from datetime import date, timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException

from api.rate_limit import enforce_daily_limit
from db.models import RateLimitLog
from db.session import SessionLocal


def test_rate_limit_resets_on_next_day():
    """The same IP receives a fresh limit on the next calendar day."""

    db = SessionLocal()
    ip_address = f"test-rate-limit-{uuid4()}"
    first_day = date(2026, 10, 13)
    next_day = first_day + timedelta(days=1)

    try:
        enforce_daily_limit(db, ip_address, first_day)
        enforce_daily_limit(db, ip_address, first_day)

        with pytest.raises(HTTPException) as error:
            enforce_daily_limit(db, ip_address, first_day)

        assert error.value.status_code == 429

        enforce_daily_limit(db, ip_address, next_day)

        row = db.get(RateLimitLog, (ip_address, next_day))
        assert row is not None
        assert row.request_count == 1

    finally:
        db.query(RateLimitLog).filter(
            RateLimitLog.ip_address == ip_address
        ).delete()
        db.commit()
        db.close()