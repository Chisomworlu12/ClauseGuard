import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agents.aggregator import (
    attach_disclaimer,
    generate_overall_verdict,
    log_verifier_decision,
    mark_escalated,
    sort_judgments,
)
from config import settings


def test_sort_judgments_by_risk_priority():
    judgments = [
        {"risk_level": "low", "category": "x", "reason": "low"},
        {"risk_level": "high", "category": "x", "reason": "high"},
        {"risk_level": "unable_to_assess", "category": "x", "reason": "unknown"},
        {"risk_level": "medium", "category": "x", "reason": "medium"},
    ]

    result = sort_judgments(judgments)

    assert [item["risk_level"] for item in result] == [
        "high",
        "medium",
        "low",
        "unable_to_assess",
    ]


def test_unknown_risk_levels_are_last():
    judgments = [
        {"risk_level": "unknown", "category": "x", "reason": "unknown"},
        {"risk_level": "high", "category": "x", "reason": "high"},
    ]

    result = sort_judgments(judgments)

    assert [item["risk_level"] for item in result] == [
        "high",
        "unknown",
    ]


def test_generate_overall_verdict_uses_all_judgments():
    judgments = [
        {
            "risk_level": "high",
            "category": "termination",
            "reason": "No notice: termination-redflag-03",
        },
        {
            "risk_level": "medium",
            "category": "payment_terms",
            "reason": "Long payment period: payment_terms-norm-01",
        },
    ]

    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
        {
            "overall_verdict": (
                "The contract presents high termination risk and moderate payment risk."
            )
        }
    )

    with patch("agents.aggregator._get_client", return_value=client):
        verdict = generate_overall_verdict(
            judgments,
            contract_context="Commercial services agreement",
        )

    assert verdict == (
        "The contract presents high termination risk and moderate payment risk."
    )

    call = client.chat.completions.create.call_args.kwargs

    assert call["model"] == "deepseek-chat"
    assert len(call["messages"]) == 2

    prompt = call["messages"][1]["content"]
    assert "termination-redflag-03" in prompt
    assert "payment_terms-norm-01" in prompt
    assert "Commercial services agreement" in prompt

def test_attach_disclaimer_always_includes_configured_disclaimer():
    verdict = "The contract presents high termination risk."

    final_report = attach_disclaimer(verdict)

    assert verdict in final_report
    assert "Disclaimer:" in final_report
    assert settings.legal_disclaimer in final_report

def test_attach_disclaimer_rejects_empty_disclaimer(monkeypatch):
    monkeypatch.setattr(settings, "legal_disclaimer", "")

    with pytest.raises(ValueError, match="legal_disclaimer must not be empty"):
        attach_disclaimer("Some verdict")

def test_mark_escalated_marks_low_confidence_judgments():
    judgments = [
        {
            "risk_level": "high",
            "confidence": "low",
            "category": "termination",
            "reason": "Ambiguous termination language.",
        },
        {
            "risk_level": "medium",
            "confidence": "normal",
            "category": "payment_terms",
            "reason": "Clear payment deadline.",
        },
        {
            "risk_level": "unable_to_assess",
            "confidence": "normal",
            "category": "other",
            "reason": "References do not settle the issue.",
        },
    ]

    result = mark_escalated(judgments)

    assert [item["confidence"] for item in result] == [
        "escalated",
        "normal",
        "escalated",
    ]


def test_log_verifier_decision_writes_verifier_audit_row():
    db = MagicMock()
    run_id = uuid.uuid4()

    judgments = [
        {
            "risk_level": "high",
            "category": "termination",
            "reason": "No notice.",
            "confidence": "normal",
        }
    ]

    audit = log_verifier_decision(
        db=db,
        run_id=run_id,
        judgments=judgments,
        overall_verdict="The contract has significant termination risk.",
        final_report="The contract has significant termination risk.\n\nDisclaimer: test",
    )

    row = db.add.call_args.args[0]

    assert row.agent == "verifier"
    assert row.run_id == run_id
    assert row.clause_id is None
    assert row.input_summary["judgment_count"] == 1
    assert row.output_summary["overall_verdict"].startswith(
        "The contract has significant"
    )
    assert row.model_used == settings.deepseek_model
    assert audit is row
    db.commit.assert_called_once()
    db.refresh.assert_called_once_with(row)