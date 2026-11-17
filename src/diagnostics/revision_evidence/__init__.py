"""Evidence-only revision experiments for the empirical graph-signal study."""

METRICS = ("ndcg@10", "ndcg@20", "recall@10", "recall@20")
JOINT_FEATURES = ("s", "log1p_item_degree", "log1p_q", "n_eff", "h", "h_minus_s", "variance")

