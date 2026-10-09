import json
import uuid

import pytest

from agents.aggregator import mark_escalated, sort_judgments
from agents.judgment import _parse_judgment


def test_judgment_output_matches_aggregator_contract():
    """Verify judgment output contains every field required by aggregation."""

    judgment = _parse_judgment(
        json.dumps(
            {
                "risk_level": "high",
                "category": "termination",
                "reason": "termination-redflag-03",
            }
        ),
        ["termination-redflag-03"],
    )

    judgment["clause_id"] = str(uuid.uuid4())
    judgment["position"] = 1

    required_fields = {
        "risk_level",
        "category",
        "reason",
        "clause_id",
        "position",
    }

    assert required_fields.issubset(judgment)
    assert judgment["risk_level"] == "high"
    assert judgment["category"] == "termination"
    assert judgment["reason"]
    assert judgment["clause_id"]
    assert judgment["position"] == 1


def test_broken_judgment_contract_is_rejected():
    """Ensure malformed judgment output cannot enter the aggregation pipeline."""

    broken_judgment = {
        "category": "termination",
        "reason": "termination-redflag-03",
    }

    with pytest.raises(ValueError, match="Unknown risk_level"):
        _parse_judgment(json.dumps(broken_judgment), ["termination-redflag-03"])

def test_aggregator_preserves_judgment_fields():
    """Verify sorting and escalation do not discard judgment data."""

    judgment = {
        "risk_level": "high",
        "category": "termination",
        "reason": "termination-redflag-03",
        "confidence": "normal",
        "clause_id": str(uuid.uuid4()),
        "position": 1,
    }

    sorted_judgments = sort_judgments([judgment])
    escalated_judgments = mark_escalated(sorted_judgments)

    assert escalated_judgments[0]["risk_level"] == "high"
    assert escalated_judgments[0]["category"] == "termination"
    assert escalated_judgments[0]["reason"] == "termination-redflag-03"
    assert escalated_judgments[0]["clause_id"] == judgment["clause_id"]
    assert escalated_judgments[0]["position"] == 1
    assert escalated_judgments[0]["confidence"] == "normal"