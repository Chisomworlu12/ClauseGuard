from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.pipeline import run_analysis
from api.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    EvalRunResponse,
    RunSummary,
)
from db.models import AnalysisRun, EvalRun
from db.session import get_db

router = APIRouter(prefix="/api", tags=["api"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_contract(
    request: AnalyzeRequest,
    db: Session = Depends(get_db),  # noqa: B008
):
    """Run the complete contract analysis pipeline."""

    run, final_report = run_analysis(
        contract_text=request.contract_text,
        db=db,
    )

    return AnalysisResponse(
        run_id=run.id,
        status=run.status,
        report=final_report,
    )


@router.get("/runs", response_model=list[RunSummary])
def list_runs(db: Session = Depends(get_db)):  # noqa: B008
    """Return all analysis runs, newest first."""

    runs = db.scalars(
        select(AnalysisRun).order_by(AnalysisRun.created_at.desc())
    ).all()

    return [
        RunSummary(
            run_id=run.id,
            status=run.status,
            contract_text=run.contract_text,
        )
        for run in runs
    ]


@router.get("/runs/{run_id}", response_model=RunSummary)
def get_run(
    run_id: UUID,
    db: Session = Depends(get_db),  # noqa: B008
):
    """Return one analysis run by its unique ID."""

    run = db.get(AnalysisRun, run_id)

    if run is None:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    return RunSummary(
        run_id=run.id,
        status=run.status,
        contract_text=run.contract_text,
    )


@router.get("/eval/latest", response_model=EvalRunResponse)
def get_latest_evaluation(db: Session = Depends(get_db)):  # noqa: B008
    """Return the most recently persisted evaluation result."""

    evaluation = db.scalars(
        select(EvalRun).order_by(EvalRun.run_at.desc())
    ).first()

    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation run found",
        )

    return EvalRunResponse(
        model=evaluation.model,
        retrieval_enabled=evaluation.retrieval_enabled,
        accuracy=evaluation.accuracy,
        mrr=evaluation.mrr,
        ndcg=evaluation.ndcg,
        commit_hash=evaluation.commit_hash,
    )