import json

import pytest

from eval.ci_gate import main


def test_ci_gate_rejects_regression(tmp_path, monkeypatch):
    baseline = {
        "accuracy": 0.6667,
        "mrr": 0.4063,
        "ndcg": 0.4457,
        "tolerance": 0.05,
    }
    current = {
        "accuracy": 0.20,
        "mrr": 0.10,
        "ndcg": 0.10,
    }

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")

    monkeypatch.setattr("eval.ci_gate.BASELINE_PATH", baseline_path)
    monkeypatch.setattr("eval.ci_gate.CURRENT_PATH", current_path)

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 1