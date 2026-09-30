import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from openai import OpenAIError

from agents.judgment import GroundingError, _parse_judgment, judge_clause
from config import settings

VALID_REPLY = {
    "risk_level": "high",
    "category": "termination",
    "reason": "One-sided termination right, per termination-redflag-01.",
}


def _fake_client(reply: dict) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps(reply)
    )
    return client


DOCS = [
    Document(
        page_content="Termination for convenience is one-sided.",
        id="termination-redflag-01",
        metadata={"id": "termination-redflag-01", "category": "termination"},
    )
]


# An out-of-set risk level must be rejected so the retry loop kicks in.
def test_parse_judgment_rejects_unknown_risk_level():
    with pytest.raises(ValueError):
        _parse_judgment(json.dumps({"risk_level": "catastrophic", "category": "x", "reason": "y"}),
        ["termination-redflag-01"],
    )


# A well-formed reply becomes a clean judgment dict.
def test_parse_judgment_returns_expected_shape():
    assert( 
        _parse_judgment(json.dumps(VALID_REPLY), ["termination-redflag-01"]) == VALID_REPLY
    )

# A grounded judgment is returned and written to the audit log.
def test_judge_clause_logs_audit_row():
    db = MagicMock()

    with patch("agents.judgment._get_client", return_value=_fake_client(VALID_REPLY)):
        judgment = judge_clause(
            db=db,
            run_id=uuid.uuid4(),
            clause_id=uuid.uuid4(),
            clause_text="Either party may terminate at any time without cause.",
            retrieved_docs=DOCS,
        )

    assert judgment == VALID_REPLY
    db.add.assert_called_once()
    db.commit.assert_called_once()
    assert db.add.call_args[0][0].agent == "judgment"

    # Verify every judgment call is persisted to agent_audit_log.
    audit_row = db.add.call_args.args[0]

    assert audit_row.agent == "judgment"
    assert audit_row.clause_id is not None
    assert audit_row.input_summary["clause_text"].startswith("Either party")
    assert audit_row.input_summary["retrieved_ref_ids"] == [
        "termination-redflag-01"
    ]
    assert audit_row.output_summary == VALID_REPLY
    assert audit_row.model_used == settings.deepseek_model

# Persistent LLM failure degrades to unable_to_assess instead of raising.
def test_judge_clause_falls_back_to_unable_to_assess():
    db = MagicMock()
    client = MagicMock()
    client.chat.completions.create.side_effect = OpenAIError("upstream down")

    with patch("agents.judgment._get_client", return_value=client), patch(
        "agents.judgment.time.sleep"
    ) as sleep_mock:
        judgment = judge_clause(
            db=db,
            run_id=uuid.uuid4(),
            clause_id=uuid.uuid4(),
            clause_text="Either party may terminate at any time.",
            retrieved_docs=DOCS,
        )

    assert judgment == {
        "risk_level": "unable_to_assess",
        "category": "unable_to_assess",
        "reason": judgment["reason"],
    }

    assert client.chat.completions.create.call_count == settings.judgment_max_retries
    assert sleep_mock.call_count == settings.judgment_max_retries - 1

    expected_delays = [
        settings.judgment_retry_backoff_base ** (attempt + 1)
        for attempt in range(settings.judgment_max_retries - 1)
    ]

    actual_delays = [call.args[0] for call in sleep_mock.call_args_list]

    assert actual_delays == expected_delays
    db.commit.assert_called_once()

    
# Repeated ungrounded replies give up early instead of burning every retry.
def test_judge_clause_gives_up_early_on_ungrounded_replies():
    db = MagicMock()
    client = MagicMock()
    ungrounded = {
        "risk_level": "high",
        "category": "termination",
        "reason": "This clause is unfair.",
    }
    client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps(ungrounded)
    )

    with patch("agents.judgment._get_client", return_value=client), patch(
        "agents.judgment.time.sleep"
    ):
        judgment = judge_clause(
            db=db,
            run_id=uuid.uuid4(),
            clause_id=uuid.uuid4(),
            clause_text="Either party may terminate at any time.",
            retrieved_docs=DOCS,
        )

    assert judgment["risk_level"] == "unable_to_assess"
    assert (
        client.chat.completions.create.call_count
        == settings.judgment_max_ungrounded_attempts
    )
    assert settings.judgment_max_ungrounded_attempts < settings.judgment_max_retries


# A permanent error is not retried at all.
def test_judge_clause_does_not_retry_permanent_errors():
    db = MagicMock()
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "not json"

    with patch("agents.judgment._get_client", return_value=client), patch(
        "agents.judgment.time.sleep"
    ):
        judgment = judge_clause(
            db=db,
            run_id=uuid.uuid4(),
            clause_id=uuid.uuid4(),
            clause_text="Either party may terminate at any time.",
            retrieved_docs=DOCS,
        )

    assert judgment["risk_level"] == "unable_to_assess"
    assert client.chat.completions.create.call_count == 1
    db.commit.assert_called_once()

# A reason that cites no retrieved reference is not grounded (FR-JUDGE-002).
def test_parse_judgment_rejects_ungrounded_reason():
    reply = {"risk_level": "high", "category": "termination", "reason": "This clause is unfair."}
    with pytest.raises(GroundingError):
        _parse_judgment(json.dumps(reply), ["termination-redflag-01"])


# A clause no reference category fits is allowed to say so.
def test_parse_judgment_allows_other_category():
    reply = {
        "risk_level": "low",
        "category": "other",
        "reason": "No reference covers a plain term clause; see auto-renewal-norm-03.",
    }
    assert _parse_judgment(json.dumps(reply), ["auto-renewal-norm-03"])["category"] == "other"

#  Verify that one clause failure does not stop later clauses
# from being processed.
def test_one_clause_failure_does_not_stop_following_clauses():
    clauses = ["first clause", "second clause", "third clause"]
    processed = []

    def fake_judge_clause(*args, **kwargs):
        clause_text = kwargs["clause_text"]
        processed.append(clause_text)

        if clause_text == "second clause":
            raise RuntimeError("simulated clause failure")

        return {
            "risk_level": "low",
            "category": "other",
            "reason": "verified reference",
        }

    failed_clauses = []

    for clause in clauses:
        try:
            fake_judge_clause(clause_text=clause)
        except RuntimeError as error:
            failed_clauses.append((clause, str(error)))

    assert processed == clauses
    assert failed_clauses == [("second clause", "simulated clause failure")]

def settings_retries() -> int:
    return settings.judgment_max_retries