# Released scalar evidence

All numerical files are CSV. Metrics without an `_x100` or `points` suffix are on the 0–1 scale; manuscript tables multiply them by 100. Standard deviations are sample standard deviations (`ddof=1`). Missing values remain missing. A source reference beginning `artifact:` is a logical identifier of an excluded source artifact, not a file bundled in this repository. Its historical path after that prefix is retained to trace provenance without revealing a home directory. References to shared candidate adapters originally named `dnr_*` do not include DNR model code or DNR results.

| Files | Scope and keys | Use |
|---|---|---|
| `studies.csv` | Declarative S1/S2/S3 study design | Table 1 |
| `s1/benchmark_seed_metrics.csv` | 3 datasets × 6 backbones × 3 seeds × Base/NPGC/Pop; exact saved NDCG/Recall | Table 2 and Section 4.1 |
| `s1/s1_candidate_recall.csv` | Same 54 coordinates; candidate recall, oracle, truth/provenance limits | Table 2 |
| `s1/s1_oracle_headroom.csv` | 162 method-coordinate rows; `(method - Base)/(oracle - Base)` | Section 4.1 recovery ratios |
| `s1/*3seed_summary.csv`, `s1/dataset_seed_macro.csv` | Recorded aggregate cross-checks | S1 means and sample SDs |
| `s1/fresh_seed_metrics.csv` | 3 datasets × NFM/DCNv2 × 3 seeds × U/W/F × Base/Pop/NPGC | Table 3 top |
| `s1/e4_*.csv` | Saved E4 scalar evidence for Amazon-Books and ML-1M U/W/F | E4 catalog completion audit |
| `s2/seed_metrics.csv` | Yelp2018/ML-1M × 3 seeds × U/W/F × Base/Pop | Table 3 bottom |
| `s3/extracted_seed_results.csv` | R0 only, training seeds 2020–2024; matched/frozen; native or identity-inclusive/trained-only policy | Raw S3 per-repeat evidence |
| `s3/homogeneous_method_summary.csv` | `scope=main_R0`; method, protocol, metric, selection policy | Table 4 and headroom sensitivity |
| `s3/repeat_intervals.csv` | Paired Student-t repeat intervals, not user-bootstrap intervals | Figure 1, protocol changes |
| `s3/transfer_classification.csv` | `main_R0`; 3 margins; separate D1 training-repeat and D2 user-bootstrap decisions | Table 5, transfer sensitivity |
| `s3/tost_holm_primary.csv` | Registered 16-hypothesis family marked by `registered_primary_family=True`; other rows are sensitivity/exploratory | Exact-size equivalence |
| `s3/selection_regret.csv`, `s3/regret_repeat_summary.csv` | Per-seed and repeat-summary source-selection regret; `REF4` and `ALL12` are separate | Section 4.3 |
| `s3/rank_agreement.csv`, `s3/rank_repeat_summary.csv`, `s3/clear_reversal.csv` | Rank and reversal scalar statistics | Supporting diagnostics |
| `s3/user_bootstrap.csv` | Summary intervals from paired user-cluster bootstrap; no user-level rows | Conditional sensitivity |
| `checks/A_npgc_equivalence.csv` | Two saved official-versus-local numerical audits | NPGC implementation check |
| `checks/C_*.csv` | Res(p) fit/evaluation scalar metrics, per-seed contrasts, paired repeat CIs, identity and partial-dependence summaries | Figure 1(c), Table 5 bottom |
| `checks/D_mostpop*.csv` | MostPop retrieval per coordinate and dataset summaries | Section 4.1 |
| `checks/list_spearman*.csv` | Saved mean within-list score/popularity Spearman, plus defined/undefined-user counts | Section 4.2 |

S3 R0 uses backbone batch 4096, residual batch 256, and PRM+PL batch 32. All R1, mixed-regime, and intermediate A5-scope statistical rows are excluded. DNR, P3, and incomplete NFM-BPR Stage E results are excluded. E4 is the completed S1 catalog evaluation, and is distinct from the excluded NFM-BPR work.

Fixed rules appear under `selection_policy=native` in method summaries, while learned methods use `identity_inclusive` or `trained_only`. Transfer tables use identity-inclusive comparisons and omit the duplicated frozen fixed-rule contrasts (`Pop-Base`, `NPGC-Base`, `Pop-NPGC`), giving 96 matched plus 40 frozen comparisons. V is a post hoc subdivision of unresolved outcomes; the stored original D1/D2 bootstrap agreement retains its original class definition.

S1/S2 vary the split and training seed together (2020–2022). S3 has five training repeats on one fixed split and candidate draw. Student-t intervals use paired per-repeat differences with df=2 or df=4 respectively. No bootstrap draw is counted as an additional training repeat.

Historical identity limits are retained in the CSVs. In particular, some S1 historical caches lack original pre-pool candidate IDs; Yelp2018 S1 F reports current reconstruction without certifying historical candidate identity. No missing historical candidate IDs, metrics, or user-level observations were inferred for this release.

Large artifacts are intentionally excluded. A full rerun writes private model checkpoints and candidate tensors in its selected work directory. Candidate-cache schemas are documented by the generating adapters and feature-cache modules: each split stores user IDs, candidate item IDs, validity masks, relevance labels, full relevant-count denominators, backbone scores, feature tensors, scaler/scale metadata, and input identity digests. Saved per-user metric arrays align by exact user ID and full relevance denominator before resampling. To regenerate bootstrap uncertainty rather than replay its scalar summaries, generate those caches/predictions through the full pipeline and supply aligned per-user metrics to the statistics source. This CSV-only release does not pretend to reconstruct excluded user-level arrays.
