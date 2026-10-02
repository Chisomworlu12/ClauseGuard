from eval.cases import EVAL_CASES
from eval.judgment_eval import score_judgment


def test_score_judgment_requires_matching_risk_and_category():
    case = EVAL_CASES[0]

    result = score_judgment(
        case,
        {
            "risk_level": case.expected_risk_level,
            "category": case.expected_category,
        },
    )

    assert result == {
        "risk_correct": True,
        "category_correct": True,
        "exact_correct": True,
    }


def test_score_judgment_detects_wrong_labels():
    case = EVAL_CASES[0]

    result = score_judgment(
        case,
        {
            "risk_level": "low",
            "category": "other",
        },
    )

    assert result["risk_correct"] is False
    assert result["category_correct"] is False
    assert result["exact_correct"] is False