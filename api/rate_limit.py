from datetime import UTC, date, datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from db.models import RateLimitLog

MAX_REQUESTS_PER_DAY = 2


def enforce_daily_limit(
    db: Session,
    ip_address: str,
    request_date: date | None = None,
) -> None:
    """Allow two analyses per IP per calendar day."""

    request_date = request_date or datetime.now(UTC).date()

    row = db.get(RateLimitLog, (ip_address, request_date))

    if row is None:
        row = RateLimitLog(
            ip_address=ip_address,
            request_date=request_date,
            request_count=0,
        )
        db.add(row)

    if row.request_count >= MAX_REQUESTS_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail="Daily analysis limit reached. Try again tomorrow.",
        )

    row.request_count += 1
    db.commit()