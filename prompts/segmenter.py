# Prompts for the Segmenter Agent's LLM fallback (FR-SEG-002).

SYSTEM_PROMPT = "You segment contracts into clauses. Reply with JSON only."

FALLBACK_PROMPT = (
    "Split the following contract into its individual clauses. "
    "Keep each clause's original wording, don't summarize or rewrite it. "
    'Respond with JSON in exactly this shape: {"clauses": ["clause text", "clause text", ...]}'
)
