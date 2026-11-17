# Released scalar evidence

Numerical results are CSV, with structured verification records in JSON. Metrics without an `_x100` or `points` suffix are on the 0–1 scale unless the file's unit column says otherwise. Manuscript tables multiply raw metrics by 100. Relative gains are percentages, not metric points. Standard deviations are sample standard deviations (`ddof=1`). Missing values remain missing. A source reference beginning `artifact:` is a logical identifier of an excluded source artifact, not a file bundled in this repository. Its historical path after that prefix is retained to trace provenance without revealing a home directory. References to shared candidate adapters originally named `dnr_*` do not include DNR model code or DNR results.

| Files | Scope and keys | Use |
|---|---|---|
| `studies.csv` | Declarative S1/S2/S3 study design | Table 1 |
| `s1/benchmark_seed_metrics.csv` | 3 datasets × 6 backbones × 3 seeds × Base/NPGC/Pop; exact saved NDCG/Recall | Table 2 and Section 4.1 |
| `s1/s1_candidate_recall.csv` | Same 54 coordinates; candidate recall, oracle, truth/provenance limits | Table 2 |
| `s1/s1_oracle_headroom.csv` | 162 method-coordinate rows; `(method - Base)/(oracle - Base)` | Retained diagnostic evidence |
| `s1/*3seed_summary.csv`, `s1/dataset_seed_macro.csv` | Recorded aggregate cross-checks | S1 means and sample SDs |
| `s1/fresh_seed_metrics.csv` | 3 datasets × NFM/DCNv2 × 3 seeds × U/W/F × Base/Pop/NPGC | Table 3 top |
| `s1/e4_*.csv` | Saved E4 scalar evidence for Amazon-Books and ML-1M U/W/F | E4 catalog completion audit |
| `s2/seed_metrics.csv` | Yelp2018/ML-1M × 3 seeds × U/W/F × Base/Pop | Table 3 bottom |
| `s3/extracted_seed_results.csv` | R0 only, training seeds 2020–2024; matched/frozen; native or identity-inclusive/trained-only policy | Raw S3 per-repeat evidence |
| `s3/homogeneous_method_summary.csv` | `scope=main_R0`; method, protocol, metric, selection policy | Table 4 and headroom sensitivity |
| `s3/repeat_intervals.csv` | Paired Student-t repeat intervals, not user-bootstrap intervals | Original S3 protocol changes |
| `s3/transfer_classification.csv` | `main_R0`; 3 margins; separate D1 training-repeat and D2 user-bootstrap decisions | Table 5, transfer sensitivity |
| `s3/tost_holm_primary.csv` | Registered 16-hypothesis family marked by `registered_primary_family=True`; other rows are sensitivity/exploratory | Exact-size equivalence |
| `s3/selection_regret.csv`, `s3/regret_repeat_summary.csv` | Per-seed and repeat-summary source-selection regret; `REF4` and `ALL12` are separate | Section 4.3 |
| `s3/rank_agreement.csv`, `s3/rank_repeat_summary.csv`, `s3/clear_reversal.csv` | Rank and reversal scalar statistics | Supporting diagnostics |
| `s3/user_bootstrap.csv` | Summary intervals from paired user-cluster bootstrap; no user-level rows | Conditional sensitivity |
| `checks/A_npgc_equivalence.csv` | Two saved official-versus-local numerical audits | NPGC implementation check |
| `checks/C_*.csv` | Res(p) fit/evaluation scalar metrics, per-seed contrasts, paired repeat CIs, identity and partial-dependence summaries | Figure 1(c) and supporting diagnostics |
| `checks/D_mostpop*.csv` | MostPop retrieval per coordinate and dataset summaries | Section 4.1 |
| `checks/list_spearman*.csv` | Saved mean within-list score/popularity Spearman, plus defined/undefined-user counts | Section 4.2 |

S3 R0 uses backbone batch 4096, residual batch 256, and PRM+PL batch 32. All R1, mixed-regime, and intermediate A5-scope statistical rows are excluded. DNR model results remain excluded. Completed NFM-BPR and MF-BCE objective/exposure comparisons, decoupled retrieval, historical full-pool evaluation, expanded pair comparisons, and candidate-draw sensitivity are under `followup/`. E4 is the completed S1 catalog evaluation, distinct from the objective/exposure comparison.

Fixed rules appear under `selection_policy=native` in method summaries, while learned methods use `identity_inclusive` or `trained_only`. Transfer tables use identity-inclusive comparisons and omit the duplicated frozen fixed-rule contrasts (`Pop-Base`, `NPGC-Base`, `Pop-NPGC`), giving 96 matched plus 40 frozen comparisons. V is a post hoc subdivision of unresolved outcomes; the stored original D1/D2 bootstrap agreement retains its original class definition.

S1/S2 vary the split and training seed together (2020–2022). S3 has five training repeats (2020–2024) on one fixed split and candidate draw. Student-t intervals use paired per-repeat differences with df=2 or df=4 respectively. The separate candidate-draw analysis holds the training checkpoint at seed 2020 and varies five candidate RNG seeds. Neither a bootstrap draw nor a candidate draw is counted as an additional training repeat.

Historical identity limits are retained in the CSVs. In particular, some S1 historical caches lack original pre-pool candidate IDs; Yelp2018 S1 F reports current reconstruction without certifying historical candidate identity. No missing historical candidate IDs, metrics, or user-level observations were inferred for this release.

Large artifacts are intentionally excluded. A full rerun writes private model checkpoints and candidate tensors in its selected work directory. Candidate-cache schemas are documented by the generating adapters and feature-cache modules: each split stores user IDs, candidate item IDs, validity masks, relevance labels, full relevant-count denominators, backbone scores, feature tensors, scaler/scale metadata, and input identity digests. Saved per-user metric arrays align by exact user ID and full relevance denominator before resampling. To regenerate bootstrap uncertainty rather than replay its scalar summaries, generate those caches/predictions through the full pipeline and supply aligned per-user metrics to the statistics source. This CSV-only release does not pretend to reconstruct excluded user-level arrays.

## Completed follow-up evidence

All files below are under `followup/`. S3 and the two added training conditions have five training repeats on one fixed split. S1 has three split/training repeats. Candidate-draw variability uses five draws from one frozen checkpoint. These uncertainty units are never pooled.

| Files | Keys and units | Repetition and interpretation |
|---|---|---|
| `E_seed_results.csv`, `E_summary.csv` | cohort, dataset, protocol, method, training_seed for raw rows. Raw NDCG/Recall/CR/oracle are 0–1. Summary `_x100` columns are points. Spearman retains its correlation scale. | Four cohorts: S3_NFM_BCE, E1_NFM_BPR, E2_MF_BCE, S3_MF_BPR. Five training seeds each. BCE versus BPR also changes negative exposure and training-row composition. |
| `E_pop_minus_base_paired.csv`, `E_transfer.csv` | cohort, dataset, protocol or transition, contrast, metric. `_raw` is 0–1, `_x100` is points. | Three paired contrasts, 95% effect intervals, 90% equivalence intervals. The paired file includes all three contrasts despite its historical name. Figure 1(a,b) selects NDCG@20 Pop−Base. |
| `DF_s3_seed_results.csv`, `DF_s3_summary.csv`, `DF_s3_paired.csv`, `DF_s3_transfer.csv` | dataset, training_seed, method, DF; source_protocol and contrast for transfer. Raw metrics 0–1; `_x100` points. | MF-BPR retrieval plus frozen NFM scoring, five training seeds. All five methods/reference rows share candidates, CR@50, and oracle. |
| `DF_frozen_learned_seed_results.csv`, `DF_frozen_learned_summary.csv`, `DF_frozen_learned.csv` | dataset, training_seed, source_protocol, method. The last file contains paired transfer comparisons, not raw scores. | U/W-selected residual masks and PRM+PL applied without fitting to DF. Source scaler, scale, and checkpoint selection are retained. Five training seeds. |
| `DF_s1_seed_results.csv`, `DF_s1_summary.csv`, `DF_s1_paired.csv` | dataset, backbone, seed, method. Summary adds metric; paired adds contrast. Raw metrics 0–1; summary/paired `_x100` points. | Six backbones, two datasets, three split/training seeds. Retriever candidates are shared across scorers for the same split. |
| `DF_checks.json`, `DF_s1_checks.json` | Structured coordinate/check records. Numeric errors and tolerances are on the raw metric scale where named. | Candidate identity, full truth, original MF F parity, source model lineage. A currently calculated digest is not evidence of a historical unrecorded digest. |
| `S1_fullpool_seed_results.csv`, `S1_fullpool_summary.csv` | dataset, backbone, seed, method. Metric columns are raw 0–1; `users`, `pool_items`, and seeds are counts. | Five backbones × two datasets × three seeds, 30 evaluable coordinates. Another 24 coordinates retain UNVERIFIABLE_INPUT and empty metrics. |
| `S1_fullpool_comparison_summary.csv`, `S1_fullpool_checks.csv` | dataset, backbone, condition; checks also seed/check. `points` columns are ×100. Relative-percent columns are percentages. | Compare full pool with backbone Top-50. Ratio of means and mean of seed ratios are separately named. Macro first averages five backbones within each seed, then computes three-seed mean/SD. |
| `P0_extended_counts.csv`, `P0_all_pairs_reversals.csv`, `P0_sign_flips.csv` | contrast_set, subset, or dataset/backbone/analysis_family/comparison/contrast. Counts are integers; effect units follow raw/x100 suffixes. | Original eight, expanded ten, and all fifteen method pairs are separate scopes. Repeated comparisons across scopes are not independent observations. |
| `P4_seed_results.csv`, `P4_draw_variance.csv`, `P4_transfer.csv` | dataset, backbone, draw_seed, protocol, method; variance adds metric, transfer adds contrast. Raw metrics 0–1; `_x100` columns points. | Frozen training seed 2020, candidate seeds 8702031–8702035. Five draw SDs and five training-seed SDs measure different uncertainty. All eight U–W classifications are reported, including unchanged ones. |

Full-pool methods can choose different top-50 subsets. Their additional CR@50 and oracle values are method-specific. DF methods reorder one fixed candidate set, so their coverage and oracle are shared. The unavailable full-pool inputs are all 18 Amazon-Books coordinates and six QNN_alpha Yelp2018/ML-1M coordinates. Missing historical candidates were not reconstructed.

## Provenance normalization

Only private absolute location prefixes were replaced with `artifact:`. Numeric CSV strings, empty cells, row order, keys, hashes, and JSON numeric values were preserved. The exact changed fields and row/value counts are listed below and in `followup/provenance_changes.csv`. JSON `[]` denotes entries of an array. A changed-row count is per field, so the counts are not a count of distinct experiments.

| File | Changed field | Rows / string values |
|---|---|---:|
| E_seed_results.csv | source_evidence | 360 |
| E_seed_results.csv | spearman_evidence | 360 |
| DF_s3_seed_results.csv | predictions_path | 50 |
| DF_frozen_learned_seed_results.csv | checkpoint_path | 180 |
| DF_frozen_learned_seed_results.csv | predictions_path | 180 |
| DF_frozen_learned_seed_results.csv | source_result_path | 180 |
| S1_fullpool_seed_results.csv | source | 90 |
| P0_all_pairs_reversals.csv | source_evidence | 4 |
| P0_all_pairs_reversals.csv | target_evidence | 4 |
| P0_sign_flips.csv | source_evidence | 110 |
| P0_sign_flips.csv | target_evidence | 110 |
| P4_draw_variance.csv | source_evidence | 208 |
| DF_checks.json | S3.coordinates.[].complete_path | 10 |
| DF_checks.json | S3.coordinates.[].feature_cache_path | 10 |
| DF_checks.json | S3.coordinates.[].fixed_pool_path | 10 |
| DF_checks.json | S3.coordinates.[].mf_F_base_result | 10 |

There are 1,876 changed path strings across 16 fields. Other imported follow-up fields are unchanged. Logical provenance references identify excluded historical artifacts and are not executable local paths. Runtime commands take explicit data and run-directory arguments instead.
