# Splits contract text into individual clauses using rule-based heading detection.
import json
import re
import uuid

from openai import OpenAI
from sqlalchemy.orm import Session

from config import settings
from db.models import AgentAuditLog
from prompts.segmenter import FALLBACK_PROMPT, SYSTEM_PROMPT

_client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

# Matches common contract section headings such as "1.", "1.1", "Section 2", and "Article III".
HEADING_PATTERNS = re.compile(
    r"^(?:\d+\.\d*\s|\d+\.\s|section\s+\d+|article\s+[ivxlc]+)",
    re.MULTILINE | re.IGNORECASE,
)


def _split_on_headings(text: str) -> list[str]:
    matches = list(HEADING_PATTERNS.finditer(text))
    if not matches:
        return []
    clauses = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause = text[start:end].strip()
        if clause:
            clauses.append(clause)
    return clauses


def _split_on_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def segment_rule_based(contract_text: str) -> list[str]:
    clauses = _split_on_headings(contract_text)
    if len(clauses) >= 2:
        return clauses
    return _split_on_paragraphs(contract_text)


def _needs_fallback(clauses: list[str]) -> bool:
    # FR-SEG-002: too few clauses, or one that's implausibly long.
    if len(clauses) < settings.segmenter_min_clauses:
        return True
    return any(len(c.split()) > settings.segmenter_max_clause_words for c in clauses)


def segment_with_llm(contract_text: str) -> list[str]:
    response = _client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{FALLBACK_PROMPT}\n\nContract: \n{contract_text}",
            },
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    return [c.strip() for c in data.get("clauses", []) if c.strip()]


def segment(contract_text: str) -> tuple[list[str], str]:
    # The one entry point the rest of the app calls. Returns which method
    # produced the result too - FR-SEG-003 requires logging that.
    clauses = segment_rule_based(contract_text)
    if not _needs_fallback(clauses):
        return clauses, "rule_based"
    return segment_with_llm(contract_text), "llm_fallback"

def run_segmenter(db: Session, run_id: uuid.UUID, contract_text: str) -> list[str]:
    clauses, method = segment(contract_text)

    db.add(
        AgentAuditLog(
            run_id=run_id,
            clause_id = None,  # Segmenter is a run-level agent, not per-clause.
            agent = "segmenter",
            input_summary = {"contract_word_count": len(contract_text.split())},
            output_summary = {"method": method, "clause_count": len(clauses)},
            model_used = settings.deepseek_model if method == "llm_fallback" else None,
        )
    )
    db.commit()

    return clauses