import json
import time
import uuid
from collections.abc import Sequence

from langchain_core.documents import Document
from openai import OpenAI, OpenAIError
from sqlalchemy.orm import Session

from config import settings
from db.models import AgentAuditLog
from prompts.judgment import JUDGMENT_PROMPT, SYSTEM_PROMPT
from retrieval.reranker import get_reranked_retriever

_client : OpenAI | None = None

RISK_LEVELS = ("low", "medium","high", "unable_to_assess")

UNABLE_TO_ASSESS = {
    "risk_level": "unable_to_assess",
    "category": "unable_to_assess",
    "reason": "unable_to_assess",
}

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key= settings.deepseek_api_key, 
            base_url= settings.deepseek_base_url
        )
    return _client

def _format_references(docs: Sequence[Document]) -> str:

    """Serialize retrieved KB documents into the JSON block the prompt expects."""

    references = [
        {
            "id" : doc.metadata.get("id") or doc.id,
            "category" : doc.metadata.get("category"),
            "reference_text" : doc.page_content,
        }
        for doc in docs
    ]
    return json.dumps(references, indent=2)


class GroundingError(ValueError):
    """The reply's reason cited none of the retrieved references."""



def _parse_judgment(content: str, reference_ids: Sequence[str]) -> dict:

    """Parse the LLM reply, rejecting unknown risk levels and ungrounded reasons."""

    data = json.loads(content)

    risk_level = str(data.get("risk_level", ""))
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"Unknown risk_level: {risk_level}")

    reason = str(data.get("reason") or "")

    # A judgment with no cited reference is not grounded (FR-JUDGE-002).
    if risk_level != "unable_to_assess" and not any(
        reference_id in reason for reference_id in reference_ids
    ):
        raise GroundingError("Reason cites none of the retrieved references")

    return {
        "risk_level" : risk_level,
        "category" : str(data.get("category") or "other"),
        "reason" :reason,
    }

def _judge_with_llm(clause_text: str, retrieved_docs: Sequence[Document], contract_context: str) -> dict:

    """One DeepSeek call. Retries with exponential backoff (FR-JUDGE-003)."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content" : JUDGMENT_PROMPT.format(
                clause_text = clause_text,
                references = _format_references(retrieved_docs),
                contract_context= contract_context or ""
            ),
        },
    ]

    reference_ids = [doc.metadata.get("id") or doc.id for doc in retrieved_docs]
    last_error :  Exception | None = None
    ungrounded_attempts = 0

    for attempt in range(settings.judgment_max_retries):
        try:
            response = _get_client().chat.completions.create(
                model= settings.deepseek_model,
                messages= messages,
                response_format= {"type": "json_object"}
            )
            return _parse_judgment(response.choices[0].message.content, reference_ids)

        # Ordered: GroundingError subclasses ValueError, so it must come first.
        except GroundingError as error:
            # The model ignored the cite-an-id instruction; the same prompt will
            # likely fail again, so give up early instead of burning retries.

            ungrounded_attempts += 1
            if ungrounded_attempts >= settings.judgment_max_ungrounded_attempts:
                return {**UNABLE_TO_ASSESS, "reason": f"unable_to_assess: {error}"}

            # Permanent: sending the same malformed request again cannot help.
            # JSONDecodeError subclasses ValueError, so a malformed reply lands here too.
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
            return {**UNABLE_TO_ASSESS, "reason": f"unable_to_assess: {error}"}

        # Transient: transport and server errors are worth retrying.
        except OpenAIError as error:
            last_error = error
            if attempt < settings.judgment_max_retries -1:
                delay = settings.judgment_retry_backoff_base ** (attempt + 1)
                time.sleep(delay)

    # Retries exhausted: unable_to_assess rather than raising, so one bad clause
    # never halts the rest of the run (FR-JUDGE-003).
    return {**UNABLE_TO_ASSESS, "reason": f"unable_to_assess: {last_error}"}
                          


def judge_clause(
        db: Session,
        run_id: uuid.UUID,
        clause_id: uuid.UUID,
        clause_text: str,
        retrieved_docs: Sequence[Document] | None = None,
        contract_context: str = "",
)-> dict:

    if retrieved_docs is None:
        retrieved_docs = get_reranked_retriever().invoke(clause_text)

    judgment = _judge_with_llm(clause_text, retrieved_docs, contract_context)

    db.add(
        AgentAuditLog(
            run_id = run_id,
            clause_id = clause_id, # Judgment is a per-clause agent.
            agent= "judgment",
            input_summary = {
                "clause_text": clause_text,
                "contract_context" : contract_context,
                "retrieved_ref_ids": [
                    doc.metadata.get("id") or doc.id for doc in retrieved_docs
                ],
            },
            output_summary =judgment,
            model_used = settings.deepseek_model
        )
    )
    db.commit()

    return judgment