"""Fail CI when evaluation metrics fall below the approved baseline."""

import json
import sys
from pathlib import Path

BASELINE_PATH = Path("eval/baseline_metrics.json")
CURRENT_PATH = Path("eval/current_metrics.json")


def load_metrics(path: Path) -> dict:
    """Load one evaluation result from JSON."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Compare current metrics with the approved baseline."""

    baseline = load_metrics(BASELINE_PATH)
    current = load_metrics(CURRENT_PATH)

    tolerance = baseline["tolerance"]
    failures = []

    for metric in ("accuracy", "mrr", "ndcg"):
        minimum = baseline[metric] * (1 - tolerance)

        if current[metric] < minimum:
            failures.append(
                f"{metric}: current={current[metric]:.4f}, "
                f"minimum={minimum:.4f}"
            )

    if failures:
        print("Evaluation regression detected:")
        print("\n".join(failures))
        sys.exit(1)

    print("Evaluation gate passed.")
    print(f"accuracy={current['accuracy']:.4f}")
    print(f"mrr={current['mrr']:.4f}")
    print(f"ndcg={current['ndcg']:.4f}")


if __name__ == "__main__":
    main()