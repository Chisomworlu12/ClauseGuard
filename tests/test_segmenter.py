from unittest.mock import patch

from agents.segmenter import _needs_fallback, segment, segment_rule_based

# Empty contracts should produce no clauses.
def test_empty_input_produces_no_clauses():
    assert segment_rule_based("") == []

# Plain text with no headings is treated as one clause.
def test_single_clause_input_with_no_structure():
    text = "This is one plain paragraph with no headings and no blank lines in it."
    result = segment_rule_based(text)
    assert result == [text]

# Too few clauses should trigger the LLM fallback.
def test_needs_fallback_below_min_clauses():
    assert _needs_fallback(["only one clause"]) is True

# A clause that is too long should trigger the LLM fallback.
def test_needs_fallback_clause_too_long():
    long_clause = "word " * 801  # over settings.segmenter_max_clause_words
    assert _needs_fallback(["short clause", long_clause]) is True

# Normal-sized clauses should stay on the rule-based path.
def test_needs_fallback_false_for_normal_input():
    assert _needs_fallback(["clause one", "clause two"]) is False

# Mock the LLM so this test does not make a real DeepSeek API call.
def test_malformed_text_triggers_llm_fallback():
    with patch("agents.segmenter.segment_with_llm") as mock_llm:
        mock_llm.return_value = ["clause a", "clause b"]
        clauses, method = segment("no headings, no blank lines, just one run-on paragraph")
        assert method == "llm_fallback"
        assert clauses == ["clause a", "clause b"]

        # Confirm the LLM fallback was actually called.
        mock_llm.assert_called_once()

# Well-structured contracts should not waste an LLM call.
def test_well_formed_text_does_not_call_llm():
    with patch("agents.segmenter.segment_with_llm") as mock_llm:
        text = "1. Termination\nEither party may terminate.\n\n2. Payment\nNet 30."
        _clauses, method = segment(text)
        assert method == "rule_based"

        # Confirm DeepSeek was not called.
        mock_llm.assert_not_called()
