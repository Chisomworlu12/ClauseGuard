"""Prompt used to synthesize the overall contract-risk verdict."""

SYSTEM_PROMPT = (
    "You are a contract-risk aggregator. Reason across all clause judgments"
     "collectively. Identify the most important risks, patterns, and uncertainties. "
    "Do not merely repeat one clause and do not use a fixed template. Return JSON only."
)

AGGREGATOR_PROMPT = """Contract context:
{contract_context}

Clause judgments, already prioritized by risk:
{judgments}

Synthesize the overall contract risk. Consider the judgments collectively, "
including high-risk clauses, repeated risk patterns, and any unable_to_assess items.

Return exactly this JSON shape:
{{"overall_verdict": "<genuine synthesized explanation>"}}"""