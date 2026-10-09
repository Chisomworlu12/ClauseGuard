import json
from pathlib import Path

import torch
from peft import PeftModel
from sqlalchemy.orm import Session
from transformers import AutoModelForCausalLM, AutoTokenizer

from agents.judgment import judge_clause
from config import settings
from db.models import AnalysisRun, Clause
from db.session import SessionLocal
from eval.cases import EVAL_CASES
from eval.persistence import persist_eval_run
from prompts.judgment import JUDGMENT_PROMPT, SYSTEM_PROMPT
from retrieval.reranker import get_reranked_retriever


def score_judgment(case, judgment: dict) -> float:
    """Return 1.0 only when both risk and category are correct."""

    return float(
        judgment.get("risk_level") == case.expected_risk_level
        and judgment.get("category") == case.expected_category
    )


def format_references(documents) -> str:
    """Convert retrieved documents into the judgment prompt format."""

    references = []

    for document in documents:
        references.append(
            {
                "id": document.metadata.get("id") or document.id,
                "category": document.metadata.get("category"),
                "text": document.page_content,
            }
        )

    return json.dumps(references, indent=2)


def parse_local_judgment(text: str) -> dict:
    """Extract the JSON judgment from the local model response."""

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Local model did not return a JSON object")

    try:
        judgment = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        return {
            "risk_level": "unable_to_assess",
            "category": "other",
            "reason": f"Local model returned invalid JSON: {error}",
        }

    required_fields = {"risk_level", "category", "reason"}

    if not required_fields.issubset(judgment):
        raise ValueError("Local model response is missing judgment fields")

    return judgment



def load_eval_cases():
    """Load the held-out cases used for fine-tuned model evaluation."""

    eval_path = Path("data/fine_tuning/eval.jsonl")

    with eval_path.open(encoding="utf-8") as file:
        eval_texts = {
            json.loads(line)["messages"][1]["content"]
            for line in file
        }

    return [
        case for case in EVAL_CASES
        if case.clause_text in eval_texts
    ]

class LocalJudge:
    """Run judgments with the fine-tuned local Qwen adapter."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.fine_tuned_base_model
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            settings.fine_tuned_base_model,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            settings.fine_tuned_adapter_path,
        )

        self.model.to(self.device)
        self.model.eval()

    def judge(
        self,
        clause_text: str,
        retrieved_documents,
        contract_context: str,
    ) -> dict:
        """Generate one judgment using the fine-tuned adapter."""

        references = format_references(retrieved_documents)

        user_prompt = JUDGMENT_PROMPT.format(
            contract_context=contract_context,
            clause_text=clause_text,
            references=references,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated_tokens = output[0][input_ids.shape[-1] :]
        response_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return parse_local_judgment(response_text)


def evaluate_frontier(
    db: Session,
    retrieval_enabled: bool,
    retriever,
    cases,
) -> float:
    """Evaluate the existing frontier judgment path."""

    run = AnalysisRun(
        contract_text="Phase 9 frontier comparison evaluation",
        status="running",
        judgment_model="frontier",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    total_score = 0.0

    for position, case in enumerate(cases, 1):
        clause = Clause(
            run_id=run.id,
            text=case.clause_text,
            position=position,
            risk_level="unable_to_assess",
            category="pending",
            reason="pending",
            confidence="normal",
        )
        db.add(clause)
        db.commit()
        db.refresh(clause)

        documents = (
            retriever.invoke(case.clause_text)
            if retrieval_enabled
            else []
        )

        judgment = judge_clause(
            db=db,
            run_id=run.id,
            clause_id=clause.id,
            clause_text=case.clause_text,
            retrieved_docs=documents,
            contract_context="Phase 9 comparison evaluation",
        )

        total_score += score_judgment(case, judgment)

    run.status = "complete"
    db.commit()

    return total_score / len(cases)


def evaluate_local(
    local_judge: LocalJudge,
    retrieval_enabled: bool,
    retriever,
    cases,
) -> float:
    """Evaluate the fine-tuned local model."""

    total_score = 0.0

    for case in cases:
        documents = (
            retriever.invoke(case.clause_text)
            if retrieval_enabled
            else []
        )

        judgment = local_judge.judge(
            clause_text=case.clause_text,
            retrieved_documents=documents,
            contract_context="Phase 9 comparison evaluation",
        )

        total_score += score_judgment(case, judgment)

    return total_score / len(cases)


def main() -> None:
    """Run and persist the complete 2x2 model comparison."""

    retriever = get_reranked_retriever()
    local_judge = LocalJudge()
    eval_cases = load_eval_cases()
    db = SessionLocal()

    print(f"held-out cases evaluated: {len(eval_cases)}")

    configurations = [
        ("frontier", True),
        ("frontier", False),
        ("fine_tuned", True),
        ("fine_tuned", False),
    ]

    try:
        for model_name, retrieval_enabled in configurations:
            if model_name == "frontier":
                accuracy = evaluate_frontier(
                    db=db,
                    retrieval_enabled=retrieval_enabled,
                    retriever=retriever,
                    cases=eval_cases,
                )
            else:
                accuracy = evaluate_local(
                    local_judge=local_judge,
                    retrieval_enabled=retrieval_enabled,
                    retriever=retriever,
                    cases=eval_cases,
                )

            persist_eval_run(
                db,
                model=model_name,
                retrieval_enabled=retrieval_enabled,
                accuracy=accuracy,
                mrr=0.0,
                ndcg=0.0,
            )

            print(
                f"model={model_name} "
                f"retrieval={retrieval_enabled} "
                f"accuracy={accuracy:.4f}"
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()