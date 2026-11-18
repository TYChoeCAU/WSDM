# Recorded experiment settings

This document summarizes the executed settings, not the upstream default YAML. Complete restored configurations remain in `configs/`. The configuration index below links every dataset/backbone/seed family, including model-specific fields, feature columns, thresholds, numerical constants, selection trials, and lineage hashes. No parameter search or experiment was performed to prepare this release.

## Configuration index

| Scope | Complete records | What takes precedence |
|---|---|---|
| S1: 54 backbone/split coordinates | [configs/s1/](configs/s1/) (`dataset/backbone/seed_2020.json`, `seed_2021.json`, `seed_2022.json`) | `backbone_config` and `npgc`; all six backbone names are retained |
| S2: 6 MF-BPR/split coordinates | [configs/s2/](configs/s2/) | `selected_profile`, full four-profile `profile_search` |
| S3: 20 backbone fits | [configs/s3/backbones/](configs/s3/backbones/), [backbone_index.json](configs/s3/backbone_index.json) | `profile`, fixed W candidate identity, selected backbone lineage |
| S3: 60 protocol-specific method bundles | [configs/s3/applied/](configs/s3/applied/) | Applied method setting and selector, including final batch sizes |
| PRM+PL selection: 12 grids × 2 trials | [configs/s3/prm_selection/](configs/s3/prm_selection/) | Recorded validation trials, selected learning rate, selection batch |
| Fixed data/candidates | [configs/s3/master/](configs/s3/master/), [manifest.json](configs/s3/manifest.json) | Split/pool identity and full relevance denominators |
| E1/E2 objective/exposure conditions and P4 | [configs/followup/declaration.json](configs/followup/declaration.json) | Five training seeds, batch 4096, candidate draw seeds |
| E3 BCE+neg.: 12 fits | [configs/e3/fixed_settings.json](configs/e3/fixed_settings.json); executable [configs/followup/e3/](configs/followup/e3/) and [e3_declaration.json](configs/followup/e3_declaration.json) | All 12 executed profiles, augmentation rules, and the actual three-seed execution budget |
| Statistics and selection policies | [configs/s3/analysis_plan.json](configs/s3/analysis_plan.json) | Contrasts, margins, bootstrap repeats, tie handling, identity policy |
| External artifact bindings | [configs/followup/manifest.example.json](configs/followup/manifest.example.json) | Template only; replace with actual local paths and lineage bindings |

Paths beginning `artifact:` or `historical/`, and historical run locations in provenance fields, identify excluded source artifacts. They are not runtime defaults. Full reproduction uses the explicit data/run arguments described in [README.md](README.md).

## Data partitions, labels, and repetition units

| Study/condition | Data split | Repeated quantity | Seeds |
|---|---|---|---|
| S1, S2 | Random 80/10/10 train/validation/test; shuffled observation rows (`RO`, `group_by=None`) | The split and model training change together | 2020, 2021, 2022 |
| S3 observed BCE/BPR, E1 NFM-BPR, E2 MF-BCE | Fixed outer seed-2020 split; each user's training observations are partitioned into backbone A (90%) and reranker B (10%) with the recorded pair-disjoint rule | Model training only; data and candidate draws are fixed | 2020, 2021, 2022, 2023, 2024 |
| E3 BCE+neg. | Same fixed S3 A/B split, validation candidates, and graph | Model training only | 2020, 2021, 2022 |
| P4 candidate sensitivity | Same S3 split; frozen training-seed-2020 model | Candidate RNG only | 8702031, 8702032, 8702033, 8702034, 8702035 |
| N1 score/popularity diagnostics | Existing stored candidate lists and native NPGC scores | Five S3 training repeats; S1 unavailable rows remain explicit | S3 2020–2024; S1 inventory 2020–2022 |

All studies label ratings of at least 4 as relevant. S1 observed-pair classification uses all retained observed rows with their thresholded labels, without synthetic training negatives. The exact side-feature columns and data filtering are preserved in the per-dataset configs. The common prepared-data filters require at least 10 observations per user/item. Binary graph edges merge repeated user–item observations and include all observed backbone-training pairs, including observed low ratings. In S3/E3, synthetic negatives never enter the A-only graph.

The A/B split is on user–item pairs rather than independently splitting duplicate observation rows. The implementation retains each pair's original observation multiplicity inside its assigned partition. B receives `floor(0.1 × unique observed items)` for each user. Users with fewer than 10 unique items stay entirely in A. If all relevant pairs would fall in B, the recorded deterministic swap preserves at least one relevant A pair. The reranker training slate cap is 89 for Yelp2018 and 25 for ML-1M, as recorded in `configs/s3/master/*_slate.json`. This cap controls B-supervision grouping, not the 50-candidate inference limit.

## Candidate rules

| Rule | Executed setting |
|---|---|
| S1 historical benchmark U | Request 100 negatives per relevant item; uniform draws with replacement followed by merging duplicate candidate IDs in the benchmark path. The resulting unique count can be lower than the request. |
| Fresh U and W | U draws uniformly; W weights items by `1 + d_i`. W matches U's number of distinct negatives. Relevant items are retained before backbone scoring. The existing sampler and its eligibility rules define the fixed draw. |
| S3 U/W | Request 100 unique negatives per relevant target, capped by eligible catalog size, with fixed candidate seed 2020. W uses the A-only binary item degree. |
| F test candidates | Every non-padding catalog item outside all user training observations and validation-relevant items. S3 training observations mean A+B. Held-out relevant test items are not forcibly inserted into F Top-50. Validation excludes training observations. |
| Top-50 and cutoff | Frozen backbone selects 50 candidates; all standard rerankers share those IDs, masks, labels, and full relevant-count denominators. Rank by score descending, then mapped item ID ascending. Main cutoff is 20; retained evidence also includes @10 metrics. |
| DF | Same-split MF-BPR retrieves catalog Top-50. S1 uses its S2 retriever, then frozen S1 scoring. S3 uses its MF-BPR retriever, then frozen NFM scoring. Source U/W rerankers retain their source checkpoint, scaler, and residual scale when applied to DF. No learned reranker is fit under DF. |
| S1 whole sampled pool | Apply each fixed rule to the retained pre-Top50 benchmark pool; each method may choose its own Top-50. This is not protocol F. Historical record order and original full relevance denominators are retained. Missing historical pools are not regenerated under the old identity. |
| Popularity | `p_i = log(1 + d_i)`, where `d_i` is binary training-graph item degree. Sorting by this score agrees with sorting by degree. MostPop retrieval from a whole pool/catalog is distinct from Pop within another retriever's list. |

The exact exclusion record accompanies each regenerated list. See [src/studies/corrected/corrected_sampling.py](src/studies/corrected/corrected_sampling.py), [src/studies/s3/cross_candidates.py](src/studies/s3/cross_candidates.py), and [src/studies/followup/s1_df.py](src/studies/followup/s1_df.py).

## S1 backbones

All six use Adam, optimizer weight decay 0, training batch 4096, validation every epoch, no gradient clipping, and validation **AUC** for selection. The original Trainer uses `stopping_step=5` with its strict `stale > stopping_step` stopping comparison. Exact ties retain the earlier selected checkpoint. Observed-pair classification loss and each model's internal regularizer remain unchanged. The optimizer's zero weight decay does not imply that all model regularizers are zero.

| Backbone | Embedding | Maximum epochs | Yelp2018 learning rate | Amazon-Books learning rate | ML-1M learning rate |
|---|---:|---:|---:|---:|---:|
| NFM | 64 | 300 | 0.0001 | 0.0005 | 0.001 |
| xDeepFM | 64 | 300 | 0.0001 | 0.0001 | 0.0001 |
| DCNV2 | 64 | 300 | 0.0005 | 0.0005 | 0.005 |
| AutoInt | 64 | 300 | 0.0005 | 0.0005 | 0.001 |
| EulerNet | 64 | 300 | 0.0001 | 0.001 | 0.001 |
| QNN_alpha | 16 | 100 | 0.001 | 0.001 | 0.001 |

These optimization values are the same across the three seeds of each dataset/backbone. Architecture settings vary by dataset as follows; list order is **Yelp2018 / Amazon-Books / ML-1M**.

| Backbone | Recorded architecture and model regularization |
|---|---|
| NFM | Hidden widths `[512,512,512]`; dropout 0.5 on all datasets. |
| xDeepFM | Hidden widths `[512,512,512] / [256,256,256] / [512,512,512]`; CIN widths `[200,200,200] / [100,100,100] / [100,100,100]`; dropout `0.1 / 0.5 / 0.2`; `direct=False`, regularizer 0.0005. |
| DCNV2 | Hidden widths `[512,512] / [768,768] / [256,256]`; cross layers `2 / 3 / 2`; dropout 0.2; regularizer `2 / 2 / 1`; parallel structure, `mixed=False`, recorded expert count 4 and low rank 128. |
| AutoInt | Hidden widths `[512,512,512] / [256,256,256] / [512,512,512]`; attention size 8, 2 layers, 4 heads; three dropout values `[0.1,0.1,0.1] / [0.2,0.2,0.2] / [0.5,0.5,0.5]`. |
| EulerNet | Order list `[64] / [16] / [7]`; explicit dropout `0.5 / 0.3 / 0.3`, implicit dropout `0.3 / 0.5 / 0.5`; `apply_norm=False`, regularizer 1e-5. |
| QNN_alpha | 1 layer, 4 rows, 4 heads, dropout 0.1, batch normalization enabled, embedding regularizer 0.01. |

The restored S1 configurations are the final selected settings used for the benchmark. The release does not claim to reconstruct an unrecorded earlier S1 model-search history. S2 and PRM+PL selection grids are explicitly recorded below.

## S2 and S3 backbone selection

| Condition | Optimizer and learning rate | Batch | Epoch budget | Validation and stopping |
|---|---|---:|---|---|
| S2 MF-BPR, both datasets | Adam, lr 0.001, weight decay 0, embedding 64 | 4096 | Maximum 500, minimum 50 | Full-catalog validation NDCG@20 every 10 epochs; patience 10 validation checks; minimum improvement 0 |
| S3 NFM, Yelp2018 | Adam, lr 0.0001, weight decay 0; embedding 64, `[512,512,512]`, dropout 0.5 | 4096 | Maximum 300, no minimum-epoch floor | Fixed W validation NDCG@20 every epoch; original strict `stale > 5` rule |
| S3 NFM, ML-1M | Same NFM settings, lr 0.001 | 4096 | Maximum 300, no minimum-epoch floor | Same fixed W selector and stopping rule |
| S3 MF-BPR, both datasets | Adam, lr 0.001, weight decay 0, embedding 64 | 4096 | Maximum 500, minimum 50 | Fixed W validation NDCG@20 every 10 epochs; stop at 10 stale checks |

No backbone gradient clipping is applied. Selection uses strict improvement and keeps the earlier checkpoint on exact ties. A single selected backbone per seed is frozen across U/W/F. The S2 seed-2020 search is the four profiles `learning_rate ∈ {0.001, 0.003}` × `weight_decay ∈ {0, 0.0001}`, with the same epoch/batch budget. Both datasets selected lr 0.001 and weight decay 0. Those profiles were then frozen for seeds 2021 and 2022. The S3 MF profile reuses these validated hyperparameters with a fixed-W selector, rather than repeating a catalog search.

## Architecture × training conditions

| Training condition | Rows and negatives | Implemented scope |
|---|---|---|
| Observed BCE | BCEWithLogits on all original A rows, positives and low-rating negatives; no additional unobserved rows | NFM baseline and E2 MF, five seeds |
| BCE+neg. | Retain every observed A row and label, append one uniform unobserved item with label 0 per A-positive row; refresh each epoch | E3 NFM and MF, three seeds |
| BPR | Pairwise BPR loss on rating≥4 A rows with one uniform unobserved negative per positive | MF baseline and E1 NFM, five seeds |

E1 NFM-BPR inherits the NFM optimizer, learning rate, architecture, validation selector, and stopping budget. E2 MF-BCE inherits the MF profile. E3 uses the same architecture-specific numerical profiles as NFM-BCE or MF-BCE. Thus the table above applies to both architectures under all three conditions. All backbone batches are 4096. The NFM-BPR implementation joins positive and negative forward batches for the existing batch normalization behavior; no new model architecture is introduced.

Training negatives exclude **all observed A+B pairs**, without consulting validation/test labels. BCE+neg. uses a dedicated generator with seed `training_seed * 10000 + epoch`, a positive-row permutation, and 4096-row rejection-sampling chunks. It has the same eligible negative distribution as E1, but does not claim bitwise-identical draws because E1 also consumes RNG for model dropout. Full details and all executed E3 profiles are in [configs/e3/fixed_settings.json](configs/e3/fixed_settings.json).

## NPGC and graph features

NPGC uses the binary observed-training graph. For each user or item co-occurrence matrix `C`, normalization uses its **row sums**: `D = diag((C·1 + 1e-8)^(-1/2))`, then `D C D`. This differs from the binary-cosine neighborhood used for residual features. The native neighborhood retains the target itself, selects `k_U = k_I = 2` (5 for Amazon-Books), then applies the original row-maximum weight normalization and weighted aggregation. The Cartesian user/item neighbor weights include the original Eq.12 first-pair correction. `alpha=1`; IDF, shrinkage, and evidence attenuation are disabled. No new neighborhood size was selected for these comparisons.

Implementation: [src/graph_rerank/graph.py](src/graph_rerank/graph.py), [src/graph_rerank/reranker.py](src/graph_rerank/reranker.py), [src/studies/corrected/native.py](src/studies/corrected/native.py). Official-versus-local score checks in [results/checks/A_npgc_equivalence.csv](results/checks/A_npgc_equivalence.csv) had maximum absolute differences `2.980232238769531e-7` and `3.5762786865234375e-7`, both below `4e-7`, on two audited ML-1M cells. Their configured numerical tolerance was 1e-3. These measured two-cell checks do not certify every coordinate.

## Residual modules

The S3 residual predicts `s + t * g(x)`. Its input order is `[s, log1p_item_degree, log1p_q, n_eff, h, h_minus_s, variance]`. Here `h` is the fixed graph prediction, `Q` the path-weight support mass, `N_eff` effective support, and variance the weighted neighbor-score dispersion. The **7→32→16→1** ReLU MLP has **801 parameters**, with zero output-layer weight/bias so the initial prediction is Base. Each coordinate is standardized using only valid fit candidates. Standard deviations and `t = std(fit candidate s)` use population SD and a 1e-6 floor. No target-protocol scaler is fit during frozen inference.

Masks act **after** standardization. Raw `s` in the residual output, scaler, and `t` are unchanged. Every configuration retains all seven input dimensions.

| Method ID | Interpretation | Mask in the input order above |
|---|---|---|
| S | Score only | `[1,0,0,0,0,0,0]` |
| P | Res(p), score and popularity | `[1,1,0,0,0,0,0]` |
| R | Score and support relations | `[1,0,1,1,0,0,0]` |
| PR | Score, popularity, support | `[1,1,1,1,0,0,0]` |
| H | Res(H), neighbor prediction and dispersion | `[1,0,0,0,1,1,1]` |
| RH | Score, support, neighbor statistics | `[1,0,1,1,1,1,1]` |
| PH | Score, popularity, neighbor statistics | `[1,1,0,0,1,1,1]` |
| PRH | All inputs | `[1,1,1,1,1,1,1]` |

The loss averages pairwise logistic positive/negative comparisons within each valid slate, then averages slates; score scaling follows the recorded implementation. AdamW uses lr **1e-3**, weight decay **1e-4**, batch **256 slates**, gradient norm clip **5**, maximum **50 epochs**, minimum **5**, patience **8**, and early-stopping-anchor `min_delta=1e-4`. Exact validation NDCG@20 under the training protocol selects between the initial identity scorer and every trained checkpoint, with earlier ties retained. The min-delta anchor controls early stopping, not replacement of the exact best checkpoint. `trained_only` results remain separate supporting evidence. Test scores do not select the checkpoint.

Mask constants are in [src/studies/s3/cross_common.py](src/studies/s3/cross_common.py); the applied method settings are in [configs/s3/applied/](configs/s3/applied/). Frozen U→F, W→F, and U/W→DF preserve source weights, scaler, scale, and selection.

## PRM+Principled Learning

Each of the 12 dataset/backbone/protocol cells selected `learning_rate ∈ {1e-4, 3e-4}` using seed 2020 validation NDCG@20, then froze it across all five training seeds. Exact selection ties choose the lower learning rate. The original loss combines three masked probability BCE terms with three rank-weighted squared-consistency terms. Inputs are user ID, candidate item ID, frozen backbone score, and initial position.

All **60 final fits** use Adam, weight decay **1e-4**, batch **32**, maximum **100 epochs**, minimum **5**, patience **8**, selection min-delta **1e-4**, gradient norm clip **5**, embedding dimension **16**, model width **64**, inner width **128**, **1** attention head, dropout **0.2**, and capacity **50**. ML-1M U uses inference batch 16384 and attention element budget 1e12. The other ten cells use inference batch 1024 and attention element budget 8e6. These inference controls do not replace the final training batch. The inherited `parent_selection_metric` field refers to an older source setting; the applied selector is validation NDCG@20.

| Dataset | Backbone | U selected lr | W selected lr | F selected lr |
|---|---|---:|---:|---:|
| yelp2018 | NFM | 0.0003 | 0.0001 | 0.0001 |
| yelp2018 | MF_BPR | 0.0003 | 0.0001 | 0.0003 |
| ml-1m | NFM | 0.0003 | 0.0003 | 0.0001 |
| ml-1m | MF_BPR | 0.0003 | 0.0001 | 0.0001 |

The ML-1M U **selection grids** for NFM and MF-BPR used batch **16384**, across their four learning-rate trials. The other ten selection grids used batch **32**. All final fits used batch 32. The selection records retain this historical difference; it is not silently described as a uniform selection-batch experiment.

## Metric, interval, and decision settings

NDCG/Recall use full relevant-item denominators and average over all eligible evaluation users, including users with no Top-50 hits. Candidate Recall@50 and oracle NDCG@20 describe the same list. Raw metrics are 0–1, manuscript values are ×100. Sample SD uses `ddof=1`. Paired Student-t 95% intervals use three split repeats for S1/S2 (`df=2`), five training repeats for S3/E1/E2 (`df=4`), or three E3 training repeats (`df=2`). P4 intervals describe five candidate draws conditional on one checkpoint, not five training fits.

The primary practical margin is **0.25 NDCG@20 points** (`delta_raw=0.0025`). T requires both source and target 95% intervals beyond the margin on the same side. R requires opposite sides beyond the margin. E requires both 90% equivalence intervals inside the margin. V requires a source 95% interval beyond the margin and a target 90% interval inside it. X contains the remaining unresolved cases. V is a post hoc descriptive subdivision. Sensitivity uses 0.1 and 0.5 points, plus 1%/2% of source/target oracle headroom. The 16 primary TOST hypotheses use Holm correction. User bootstrap uses deterministic seed 8702026 and 2,000 paired user-cluster resamples per seed and is reported separately from repeat variation.

The fixed eight-contrast analysis yields 96 matched plus 40 frozen comparisons after removing duplicate frozen fixed-rule contrasts. P0's ten-pair and fifteen-pair scopes are supporting extensions, not independent extra experiments. Headroom recovery averages per-seed ratios. Dataset macros average the specified backbones equally within seed before computing seed-level mean/SD. Full definitions are in [configs/s3/analysis_plan.json](configs/s3/analysis_plan.json).
