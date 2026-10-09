import json
import random
from pathlib import Path

from eval.cases import EVAL_CASES


SYSTEM_PROMPT = (
    "You are a contract-risk judgment assistant. "
    "Return JSON with risk_level, category, and reason."
)


def format_case(case) -> dict:
    """Convert one EvalCase into a chat fine-tuning example."""

    if case.ambiguous:
        reason = (
            "The available references do not provide enough information "
            "to assess this clause confidently."
        )
    else:
        references = ", ".join(case.expected_reference_ids)
        reason = f"Assessment grounded in these references: {references}."

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case.clause_text},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "risk_level": case.expected_risk_level,
                        "category": case.expected_category,
                        "reason": reason,
                    }
                ),
            },
        ]
    }


def build_dataset(
    output_dir: Path,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Create deterministic train.jsonl and eval.jsonl files."""

    cases = list(EVAL_CASES)
    random.Random(seed).shuffle(cases)

    eval_count = max(1, round(len(cases) * eval_fraction))
    eval_cases = cases[:eval_count]
    train_cases = cases[eval_count:]

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, selected_cases in (
        ("train.jsonl", train_cases),
        ("eval.jsonl", eval_cases),
    ):
        path = output_dir / filename

        with path.open("w", encoding="utf-8") as file:
            for case in selected_cases:
                file.write(json.dumps(format_case(case)) + "\n")

    assert not {case.case_id for case in train_cases} & {
        case.case_id for case in eval_cases
    }

    print(f"train cases: {len(train_cases)}")
    print(f"eval cases: {len(eval_cases)}")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    build_dataset(Path("data/fine_tuning"))