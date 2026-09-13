# Seeds the contract-risk knowledge base from JSON into the PostgreSQL database.
import json
from pathlib import Path

from db.models import ReferenceKbEntry
from db.session import SessionLocal

KB_PATH = Path(__file__).parent.parent / "data" / "kb_entries.json"


def seed():
    entries = json.loads(KB_PATH.read_text())
    db = SessionLocal()
    try:
        for e in entries:
            # merge() = insert or update by primary key, so reruns don't duplicate.
            db.merge(ReferenceKbEntry(**e))
        db.commit()
        print(f"seeded {len(entries)} entries")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
