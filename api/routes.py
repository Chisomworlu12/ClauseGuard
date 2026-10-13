import secrets
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from api.rate_limit import enforce_daily_limit
from api.schemas import (
    AdminMetricsResponse,
    AnalysisResponse,
    AnalyzeRequest,
    EvalRunResponse,
    RunSummary,
)
from config import settings
from db.models import AnalysisRun, Clause, EvalRun, RateLimitLog
from db.session import get_db

router = APIRouter(prefix="/api", tags=["api"])

def require_admin(request: Request) -> None:
    """Require the private dashboard token without returning sensitive data."""

    expected_token = settings.admin_access_token
    authorization = request.headers.get("Authorization", "")

    if not expected_token or not secrets.compare_digest(
        authorization, f"Bearer {expected_token}"
    ):
        raise HTTPException(status_code=401, detail="Admin authentication required")


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_contract(
    request: Request,
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),  # noqa: B008
):
    """Run the complete contract analysis pipeline."""

    client_ip = request.client.host if request.client else "unknown"

    enforce_daily_limit(
        db=db,
        ip_address=client_ip,
    )

    from api.pipeline import run_analysis

    run, final_report = run_analysis(
        contract_text=payload.contract_text,
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
        overall_verdict=run.overall_verdict,
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
        overall_verdict=run.overall_verdict,
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

@router.get("/admin/metrics", response_model=AdminMetricsResponse)
def get_admin_metrics(
    db: Session = Depends(get_db),  # noqa: B008
    _: None = Depends(require_admin),
):
    """Return aggregate usage metrics for the private admin dashboard."""

    today = datetime.now(UTC).date()
    total_analyses = db.scalar(select(func.count()).select_from(AnalysisRun)) or 0
    analyses_today = db.scalar(
        select(func.count())
        .select_from(AnalysisRun)
        .where(func.date(AnalysisRun.created_at) == today)
    ) or 0
    unique_clients = db.scalar(
        select(func.count(distinct(RateLimitLog.ip_address)))
    ) or 0
    completed_analyses = db.scalar(
        select(func.count())
        .select_from(AnalysisRun)
        .where(AnalysisRun.status == "complete")
    ) or 0
    failed_analyses = db.scalar(
        select(func.count())
        .select_from(AnalysisRun)
        .where(AnalysisRun.status == "failed")
    ) or 0
    risk_counts = {
        risk_level: int(count)
        for risk_level, count in db.execute(
            select(Clause.risk_level, func.count()).group_by(Clause.risk_level)
        ).all()
    }

    return AdminMetricsResponse(
        total_analyses=int(total_analyses),
        analyses_today=int(analyses_today),
        unique_clients=int(unique_clients),
        completed_analyses=int(completed_analyses),
        failed_analyses=int(failed_analyses),
        risk_counts=risk_counts,
    )