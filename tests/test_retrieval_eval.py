from eval.retrieval_eval import ndcg, reciprocal_rank


def test_reciprocal_rank_returns_first_relevant_position():
    retrieved_ids = ["wrong-id", "termination-redflag-03"]

    assert reciprocal_rank(
        retrieved_ids,
        ("termination-redflag-03",),
    ) == 0.5


def test_reciprocal_rank_returns_zero_when_missing():
    assert reciprocal_rank(
        ["wrong-id"],
        ("termination-redflag-03",),
    ) == 0.0


def test_ndcg_returns_one_for_perfect_order():
    assert ndcg(
        ["termination-redflag-03", "termination-norm-01"],
        ("termination-redflag-03", "termination-norm-01"),
    ) == 1.0