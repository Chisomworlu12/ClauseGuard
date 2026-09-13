from agents.segmenter import run_segmenter
from db.models import AnalysisRun
from db.session import SessionLocal

SAMPLES = {
    "numbered_sections": """1. Term
This Agreement begins on the Effective Date and continues for twelve months.

2. Termination
Either party may terminate this Agreement upon 30 days written notice.

3. Payment
Client shall pay Provider within thirty days of receipt of invoice.""",
    "section_headings": """Section 1. Scope of Services
The Provider agrees to perform the services described in Exhibit A.

Section 2. Confidentiality
Each party agrees to keep confidential all non-public information disclosed by the other party.

Section 3. Governing Law
This Agreement shall be governed by the laws of the State of Delaware.""",
    "article_headings": """ARTICLE I. Definitions
"Confidential Information" means any non-public information disclosed by either party.

ARTICLE II. Term and Termination
This Agreement remains in effect for two years unless terminated earlier under this Article.

ARTICLE III. Indemnification
Each party shall indemnify the other against claims arising from its own negligence.""",
    "messy_prose_no_headings": (
        "This agreement is entered into by the Client and the Consultant for consulting "
        "services as mutually agreed. The Consultant will invoice monthly and payment is "
        "due upon receipt. Either party can walk away from this deal at any time without "
        "giving a reason. All work product created belongs entirely to the Consultant "
        "regardless of who paid for it."
    ),
    "one_clause_too_long": (
        "1. Term\nThis Agreement continues for one year.\n\n2. Miscellaneous\n"
        + (
            "This provision covers all remaining matters not otherwise addressed in this "
            "agreement, including interpretation, waiver, and enforcement. "
            * 60
        )
    ),
}


def main():
    db = SessionLocal()
    for name, text in SAMPLES.items():
        run = AnalysisRun(contract_text=text, status="running", judgment_model="frontier")
        db.add(run)
        db.commit()
        db.refresh(run)

        clauses = run_segmenter(db, run.id, text)
        print(f"{name}: {len(clauses)} clauses, run_id={run.id}")


if __name__ == "__main__":
    main()
