import subprocess

from db.models import EvalRun


def current_commit_hash() -> str:

    """Return the Git commit used for the evaluation."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text = True
    ).strip()

def persist_eval_run(
        db,
        *,
        model: str,
        retrieval_enabled: bool,
        accuracy: float,
        mrr: float,
        ndcg: float,
) -> EvalRun:
    """Persist one evaluation result with its Git commit hash."""

    eval_run = EvalRun(
        model = model,
        retrieval_enabled = retrieval_enabled,
        accuracy = accuracy,
        mrr = mrr,
        ndcg = ndcg,
        commit_hash= current_commit_hash(),
    )

    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)

    return eval_run