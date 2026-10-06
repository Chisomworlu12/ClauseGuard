from eval.cases import EVAL_CASES


def test_eval_dataset_has_expected_size():
    assert 25 <= len(EVAL_CASES) <= 30


def test_eval_dataset_covers_at_least_five_categories():
    categories = {case.expected_category for case in EVAL_CASES}

    assert len(categories) >= 5


def test_eval_dataset_contains_ambiguous_cases():
    assert any(case.ambiguous for case in EVAL_CASES)


def test_every_case_has_required_labels():
    for case in EVAL_CASES:
        assert case.case_id
        assert case.clause_text
        assert case.expected_risk_level
        assert case.expected_category
        assert isinstance(case.expected_reference_ids, tuple)