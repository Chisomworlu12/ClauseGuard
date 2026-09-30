"""Aggregation helpers for prioritizing clause judgments."""

import json
import uuid

from openai import OpenAI
from sqlalchemy.orm import Session

from config import settings
from db.models import AgentAuditLog
from prompts.aggregator import AGGREGATOR_PROMPT, SYSTEM_PROMPT


def sort_judgments(judgments: list[dict]) -> list[dict]:
    # Use configurable risk priority.
    # Unknown risk levels are placed last.
    risk_order = {
        risk: position
        for position, risk in enumerate(settings.aggregator_risk_order)
    }

    fallback_position = len(risk_order)

    return sorted(
        judgments,
        key= lambda judgment: risk_order.get(
            judgment.get("risk_level"),
            fallback_position
        ),
    )

def mark_escalated(judgments: list[dict])-> list[dict]:

    low_confidence_values = set(settings.aggregator_low_confidence_values)
    escalate_risk_levels = set(settings.aggregator_escalate_risk_levels)

    escalated_judgments = []

    for judgment in judgments:
        confidence = judgment.get("confidence", "normal")

        should_escalate = (
            confidence in low_confidence_values
            or judgment.get("risk_level") in escalate_risk_levels
        )

        escalated_judgments.append(
            {
                **judgment,
                "confidence": "escalated" if should_escalate else "normal",
            }
        )

    return escalated_judgments

def _get_client() -> OpenAI:
    return OpenAI(
        api_key= settings.deepseek_api_key,
        base_url= settings.deepseek_base_url,
    )

def generate_overall_verdict(judgments: list[dict],contract_context: str = "",) -> str:
    """Generate one collective LLM verdict across all clause judgments."""

    prioritized_judgements = sort_judgments(judgments)

    response = _get_client().chat.completions.create(
        model= settings.deepseek_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": AGGREGATOR_PROMPT.format(
                    contract_context= contract_context,
                    judgments = json.dumps(
                        prioritized_judgements,
                        indent= 2
                    ),
                ),
            },
        ],
        response_format={"type": "json_object"}
    )

    data= json.loads(response.choices[0].message.content)
    verdict = str(data.get("overall_verdict") or " ").strip()

    if not verdict:
        raise ValueError("LLM returned an empty overall verdict")

    return verdict

def attach_disclaimer(overall_verdict: str) -> str:

    """Always append the configured legal disclaimer to the final verdict."""

    disclaimer = settings.legal_disclaimer.strip()

    if not disclaimer:
        raise ValueError("legal_disclaimer must not be empty")

    return f"{overall_verdict.strip()}\n\nDisclaimer: {disclaimer}"

def build_final_report(
    db: Session,
    run_id: uuid.UUID,
    judgments: list[dict],
    contract_context: str = "",
) -> str:
    """Build and audit the complete prioritized verifier report."""

    prioritized_judgments = mark_escalated(
        sort_judgments(judgments)
    )

    verdict = generate_overall_verdict(
        prioritized_judgments,
        contract_context=contract_context,
    )

    disclaimer = attach_disclaimer(verdict)

    findings = "\n\n".join(
        (
            f"Clause {judgment.get('position', index)}\n"
            f"Risk: {judgment.get('risk_level')}\n"
            f"Confidence: {judgment.get('confidence')}\n"
            f"Category: {judgment.get('category')}\n"
            f"Reason: {judgment.get('reason')}"
        )
        for index, judgment in enumerate(prioritized_judgments, 1)
    )

    final_report = (
        "PRIORITIZED FINDINGS\n\n"
        f"{findings}\n\n"
        "OVERALL VERDICT\n\n"
        f"{disclaimer}"
    )

    log_verifier_decision(
        db=db,
        run_id=run_id,
        judgments=prioritized_judgments,
        overall_verdict=verdict,
        final_report=final_report,
    )

    return final_report


def log_verifier_decision(
        db: Session,
        run_id: uuid.UUID,
        judgments: list[dict],
        overall_verdict: str,
        final_report: str
) -> AgentAuditLog:

    audit = AgentAuditLog(
        run_id = run_id,
        clause_id = None,
        agent = "verifier", 
        input_summary = {
            "judgment_count": len(judgments),
            "judgments" : judgments
        },
        output_summary = {
            "overall_verdict" : overall_verdict,
            "final_report" : final_report
        },
        model_used = settings.deepseek_model
    )

    db.add(audit)
    db.commit()
    db.refresh(audit)

    return audit