"""Small learned fusion modules over frozen graph predictions."""

METHODS = ("learned_graph_gate", "learned_graph_residual", "score_only_residual")
METRICS = ("ndcg@10", "ndcg@20", "recall@10", "recall@20")
