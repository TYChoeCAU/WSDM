
## What is included

This repository contains the small recorded evidence, the code that produced it, and portable entrypoints for new runs. It does not contain datasets, checkpoints, candidate tensors, or per-user prediction arrays. No new training was performed to prepare this release. Numerical reproduction from bundled CSVs was tested in a fresh virtual environment. Full training entrypoints were checked through imports, configuration restoration, and data preparation checks, but the complete training suite was not rerun during packaging.

| Manuscript element | Evidence | Reproduction |
|---|---|---|
| Table 1: study design | `results/studies.csv` | `paper/make_tables.py` |
| Table 2: S1 benchmark | `results/s1/benchmark_seed_metrics.csv`, `s1_candidate_recall.csv` | `paper/make_tables.py`; `scripts/run_s1.py benchmark` |
| Table 3: S1 fresh pools/catalog, S2 frozen MF | `results/s1/fresh_seed_metrics.csv`, `results/s2/seed_metrics.csv` | `paper/make_tables.py`; S1 `fresh`, S2 `evaluate` |
| Table 4: S3 matched methods | `results/s3/homogeneous_method_summary.csv` | `paper/make_tables.py`; `scripts/analyze_s3.py` |
| Table 5: T/R/E/V/X transfer counts | `results/s3/transfer_classification.csv` | `paper/transfer_counts.py`, `paper/make_tables.py` |
| Figure 1(a,b): architecture × training 2×2 | `results/followup/E_pop_minus_base_paired.csv`, `E_seed_results.csv` | `paper/make_transfer_fig.py`, `paper/verify_figure.py` |
| Figure 1(c): frozen Res(p) | `results/checks/C_paired_repeat_ci.csv`, `C_paired_seed_differences.csv` | `paper/make_transfer_fig.py`, `paper/verify_figure.py` |
| S1 whole-pool comparison | `results/followup/S1_fullpool_*.csv` | `paper/paper_numbers.py`; follow-up `fullpool` |
| Decoupled retrieval: S1/S3 DF | `results/followup/DF_*.csv`, `DF_checks.json` | `paper/paper_numbers.py`; follow-up `df-s1`, `df-s3` |
| Expanded and all-pairs transfer counts | `results/followup/P0_*.csv` | `paper/paper_numbers.py`; follow-up `p0` |
| Candidate-draw variability | `results/followup/P4_*.csv` | `paper/paper_numbers.py`; follow-up `p4` |
| Sections 4.1–4.3 and Discussion numerical claims | S1–S3, follow-up scalar evidence, MostPop, identity and sensitivity CSVs | `paper/paper_numbers.py` |

See [results/FORMAT.md](results/FORMAT.md) for keys, units, uncertainty levels, and provenance limitations. `artifact:` and `historical/` references are logical provenance identifiers, not required local paths.

## Quick reproduction without training

Run from this repository's root. Python 3.10 is supported by the pinned analysis environment. No GPU or upstream code is required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR="$PWD/.mplconfig"
python paper/make_tables.py --results results --out outputs/tables
python paper/make_transfer_fig.py --followup results/followup --checks results/checks --out outputs/transfer_fig
python paper/transfer_counts.py --run results/s3
python paper/paper_numbers.py --results results --out outputs/PAPER_NUMBER_CHECK.md
python paper/verify_figure.py --results results --out outputs/figure_value_check.json
```

The commands take seconds after installation on a typical CPU. Allow a few minutes for dependency installation. Table bodies use `booktabs` and the manuscript's `\Score` macro. The number checker writes both Markdown and JSON reports, then exits 1 if a manuscript discrepancy remains.

Expected values from the manuscript are stored in `configs/analysis_expected.json`. The checker covers all numeric cells in Tables 2–5, the retained body claims, and the added follow-up claims. It reports each mismatch without changing evidence. Numeric result values remain unchanged; private provenance paths are normalized as documented in `results/FORMAT.md`. See `paper/REPRODUCTION_STATUS.md` for the recorded check counts and validation scope.

One manuscript-setting discrepancy remains: the final S3 PRM+PL fits use batch 32, but the recorded learning-rate selection trials for ML-1M U with NFM and MF-BPR used batch 16384. The manuscript's batch-32 statement does not distinguish these two selection grids. `paper_numbers.py` intentionally exits 1 for this discrepancy. Neither recorded settings nor result values were changed to make the check pass.

Check the 27 plotted effects and their confidence intervals directly from the CSVs:

```bash
python paper/verify_figure.py --results results --out outputs/figure_value_check.json
```

Fonts and rendering environments can produce different pixels, while the figure values agree with the CSV evidence.

## Full reproduction

Use Linux, Python 3.10, a CUDA-capable GPU, and the pinned training environment. The recorded environment used PyTorch 2.5.1 with CUDA 12.1. A 24 GB GPU is suitable for the original R0 settings. More host RAM and disk space are needed for full-catalog scores, graph intermediates, and per-user predictions. Use a dedicated output directory. Do not run all stages concurrently against the same coordinate.

```bash
python -m pip install -r requirements-training.txt
python scripts/setup_third_party.py
export DATA_ROOT="$PWD/data"
export MPLCONFIGDIR="$PWD/.mplconfig"
python scripts/prepare_data.py --archive-root downloads --data-root "$DATA_ROOT"

# S1: 54 independently trained benchmark backbones, then their frozen evaluations.
python scripts/run_s1.py all --data-root "$DATA_ROOT" --out runs/s1 --device cuda:0

# S2: recorded selected MF-BPR settings, three pipeline repeats per dataset.
python scripts/run_s2.py all --data-root "$DATA_ROOT" --out runs/s2 --device cuda:0

# S3: one fixed split and candidate draw, five training repeats, R0 only.
python scripts/run_s3.py --stage all --data-root "$DATA_ROOT" --run runs/s3 --device cuda:0

# Recompute statistics and diagnostics from the newly generated artifacts.
python scripts/analyze_s3.py --run runs/s3
python scripts/run_diagnostics.py all --data-root "$DATA_ROOT" --s1-run runs/s1 --s2-run runs/s2 --s3-run runs/s3 --out runs/checks
```

Download the processed `yelp2018.zip` and `amazon-books.zip` archives linked by the [official NPGC repository](https://github.com/zyouyang/RecSys2025_NonParamGC). Download `ml-1m.zip` from [GroupLens](https://grouplens.org/datasets/movielens/1m/). Put them in `downloads/`. The preparer checks recorded archive SHA-256 values, converts raw MovieLens fields, and normalizes the Yelp date field. It does not invent ratings or side features. The checksums and supported atomic-file input paths are in `src/data/prepare.py`. A different archive requires investigation rather than silently accepting a different dataset.

`setup_third_party.py` fetches the exact NPGC commit documented in [THIRD_PARTY.md](THIRD_PARTY.md), then applies the bundled patches. To use an existing **patched** checkout, set `NPGC_ROOT` to it. The official code is not bundled here.

`run_diagnostics.py` exposes `tables`, `identity`, `spearman`, `mostpop`, `equivalence`, and `partial-dependence` stages. Equivalence and partial-dependence require regenerated models/caches and run inference without training. The other stages read saved artifacts or compute fixed popularity scores. They do not recover per-user data from the released scalar CSVs.

Stages can be run separately. S1 accepts `train`, `benchmark`, `fresh`, `mostpop`, and `summarize`, with optional `--dataset`, `--backbone`, and `--seed`. S2 accepts `prepare`, `train`, `evaluate`, and `summarize`. S2's `--search` repeats the original four-profile seed-2020 validation selection; without it, the recorded selected settings are used. S3 accepts `split`, `pools`, `backbones`, `candidates`, `matched`, `frozen`, and `statistics`, with dataset, backbone, seed, protocol, and method filters. Its default `--stage dry-run` checks the configuration inventory without launching training or inference.

For example, inspect the plan and then run a single S3 coordinate:

```bash
python scripts/run_s3.py --stage dry-run --data-root "$DATA_ROOT"
python scripts/run_s3.py --stage all --data-root "$DATA_ROOT" --run runs/s3_small --datasets ml-1m --backbones NFM --seeds 2020 --device cuda:0
```

A filtered run is a partial result, not a reproduction of the five-repeat tables. Its `all` stage records missing cohort coordinates and defers the full-cohort statistics. Fixed dataset and pool digests are checked before reuse. Fresh checkpoints need not be byte-identical to the archived ones across hardware or library changes. Their lineage and selected settings are recorded with new outputs. The released results remain separate from regenerated outputs.

**Time planning:** the CSV-only route is the verified fast route. Full reproduction is a multi-day workload on a single GPU, not a short installation test. A single small ML-1M coordinate can take minutes to hours, while Yelp/Amazon catalog scoring and some S1 backbones can dominate for many hours. Allow several days to weeks for the complete S1–S3 suite on one GPU. Recorded R0 reranker wall-time sums were approximately 1.79 hours for 60 PRM+PL fits, 0.36 hours for 480 residual fits, and 0.05 hours for 360 frozen evaluations, excluding backbone training and candidate/graph preparation. The broad full-suite range is a conservative planning estimate, not a benchmark of the extracted package. Actual time depends strongly on catalog size, CPU graph preparation, storage, and early stopping. Two GPUs can run disjoint coordinates with separate output ownership. R0 batch sizes must stay fixed for the reported experiment.

## Protocol and model settings

* **S1** uses six backbones: NFM, xDeepFM, DCNV2, AutoInt, EulerNet, and QNN_alpha. Final recorded configurations are under `configs/s1/` for three datasets and seeds 2020–2022. The benchmark requests 100 sampled negatives per relevant item, draws with replacement, then deduplicates. Fresh U/W comparisons match the target unique candidate count. S1 F uses the eligible catalog with the same exclusion rules as its fresh evaluation. NFM/DCNV2 F includes the completed Amazon-Books and ML-1M extension. MostPop is also evaluated as a retriever, separately from popularity sorting within another retriever's Top-50.
* **S2** freezes the selected MF-BPR model for U/W/F. The three repeats vary the data/training seed. `configs/s2/` records the six final settings. Its catalog-validation selection is distinct from S1's classification/AUC selection.
* **S3** uses NFM and MF-BPR on Yelp2018 and ML-1M. Seeds 2020–2024 are training repeats on one fixed split and fixed candidate draw. U/W request 100 unique negatives per relevant item. F scores eligible catalog items. Candidate IDs, labels, validity masks, and full positive-count denominators are preserved for aligned evaluation. Top-50 ties use item ID. R0 training batches are 4096 for backbones, 256 for residual modules, and 32 for PRM+PL. The 20 backbone settings and 60 protocol-specific applied bundles are under `configs/s3/`. Completed follow-ups add NFM-BPR and MF-BCE to form the architecture × training 2×2, using the recorded original optimizers and selectors.
* **NPGC** uses k=2, except k=5 on Amazon-Books. The native co-occurrence normalization and self-neighbor treatment differ from residual graph features and are not substituted for each other. `results/checks/A_npgc_equivalence.csv` records the two actual local-versus-official checks with tolerance 1e-3. Those two checks do not certify every possible coordinate.
* **Residuals** use inputs `[s, log1p_item_degree, log1p_q, n_eff, h, h_minus_s, variance]`, a 7→32→16→1 MLP, and a zero-initialized output layer. The prediction is `s + t * f(z)`. Training-set mean/std define `z`. The eight S/P/R/PR/H/RH/PH/PRH masks zero inactive coordinates after standardization. The scale `t` is the training candidate score standard deviation with the recorded floor. AdamW uses lr=1e-3, weight_decay=1e-4, clip norm 5, at most 50 epochs, minimum 5 epochs, patience 8, and min_delta=1e-4. The group-averaged score-scaled pairwise loss is preserved. Identity-inclusive and trained-only checkpoint policies remain separate. Frozen evaluation retains the source model, scaler, and scale.
* **PRM+PL** uses the recorded selected learning rates from the original `{1e-4, 3e-4}` validation comparison. Full runs inherit the coordinate's final configuration rather than perform a new test-informed search. Batch size is 32 and the maximum is 100 epochs. S3 selection uses validation NDCG@20. Backbone and reranker selection are separate.

The actual JSON settings and executable paths take precedence over these summaries. No new split, feature definition, loss, or hyperparameter budget was introduced for this release.

## Follow-up execution paths

`scripts/run_followup.py` exposes the code that generated the additional evidence. This release preparation checked imports and command help only. It did not rerun these training or inference stages. The bundled CSVs are sufficient for the quick paper commands, but are not substitutes for data, selected checkpoints, full candidate IDs, features, or predictions.

| Task | Execution | Required inputs |
|---|---|---|
| `e1`, `e2` | Train NFM-BPR or MF-BCE, evaluate frozen Base/Pop/NPGC, or aggregate completed runs | Prepared data, S3 fixed split/pools, recorded settings in `configs/followup/declaration.json` |
| `df-s3` | MF-BPR retrieval with NFM scoring, then optional frozen source rerankers | S3 selected backbone checkpoints, MF F candidates, U/W reranker checkpoints, source scaler and scale |
| `df-s1` | Six S1 scorers on same-split S2 MF candidates | Explicit S1/S2 checkpoint, split, candidate, neighbor and lineage entries in a local manifest |
| `fullpool` | Score stored pre-Top50 pools with Base, NPGC, Pop | Real historical or newly retained full pool IDs and scores, labels, full truth, selected backbone and native neighbor table |
| `p0` | Expanded and all-pairs transfer aggregation | Per-repeat S3 numeric result CSVs |
| `p4` | Five candidate RNG draws with a frozen seed-2020 checkpoint | S3 fixed split, selected checkpoint, original sampler inputs; no new model fitting |

`--help` lists stages and coordinate filters. `--stage aggregate` reads completed outputs. `e1` and `e2` with `--stage train` or `all` perform training, while DF, full-pool, and candidate-draw tasks perform inference. Run those commands only when the required large inputs have been prepared.

For example, check imports without executing experiments, then explicitly run one prepared DF coordinate:

```bash
python scripts/run_followup.py df-s3 --import-check
python scripts/run_followup.py df-s3 --dataset ml-1m --seed 2020 --phase all --data-root "$DATA_ROOT" --s3-run runs/s3 --out runs/followup --device cuda:0
```

For S1/S2 inputs, copy `configs/followup/manifest.example.json` to a local work file and fill in actual paths and SHA bindings. Paths resolve relative to that manifest. `artifact:` and `historical/` values in released evidence are provenance labels and cannot be loaded as files. Example paths identify required artifact kinds; the package does not bundle those files. Existing public S1/S2 output layouts are supported through explicit manifest entries. Never substitute a reconstructed pool for a missing historical pool while retaining its historical identity claim.

```bash
python scripts/run_followup.py df-s1 --dataset ml-1m --backbone NFM --seed 2020 --data-root "$DATA_ROOT" --manifest local_followup_manifest.json --out runs/followup --device cuda:0
python scripts/run_followup.py e1 --stage all --dataset ml-1m --seed 2020 --data-root "$DATA_ROOT" --s3-run runs/s3 --out runs/followup --device cuda:0
```

These examples run one coordinate. The reported S3 follow-up cohort requires both datasets and training seeds 2020–2024. S1 uses split seeds 2020–2022. Candidate-draw tasks fix training seed 2020 and vary draw seeds 8702031–8702035. Partial cohorts must retain their actual repeat counts. The separately recorded E1/E2 training batches come from their declaration and are not inferred from the original S3 backbone batch summary above.

## Statistics and interpretation

The primary margin is **0.25 NDCG@20 points**, equivalent to 0.0025 on the 0–1 metric scale. Paired training-repeat contrasts use Student-t 95% effect intervals and 90% equivalence intervals. A transfer requires both 95% intervals to be beyond the margin with the same sign. A reversal requires the 95% intervals to have opposite signs beyond the margin. Equivalence requires both 90% intervals to lie inside the margin. Other comparisons are unresolved. The displayed **vanish** category requires a source 95% interval beyond the margin and a target 90% interval inside the margin. It subdivides unresolved outcomes after inspecting outcomes and is not a preregistered independent test.

The 16 primary TOST hypotheses test protocol changes in two method contrasts, Pop−Base and PRM+PL−Res(p), over four dataset/backbone cells and two transitions, F−U and F−W. Holm adjustment applies to this family. A same-sign transfer does not imply equality of effect sizes. The 136 comparisons comprise 96 matched plus 40 frozen comparisons after removing duplicate fixed-rule frozen contrasts. Seed-wise paired user bootstrap uses 2,000 resamples and is reported separately from variation across training repeats. Five seeds are not 2,000 independent repetitions. `configs/s3/analysis_plan.json` records margins, tie handling, identity policy, and the fact that this analysis plan followed earlier outcome inspection.

## Scope, limitations, and licensing

DNR model code, DNR results, and R1 large-batch results are excluded. The historical directory name `baselines/dnr` contains only shared data/backbone adapters required for candidate generation, not the DNR method. Completed S1 E4 catalog evidence and the requested follow-up results are included. S3 does not establish uncertainty over alternative data splits. Five fresh candidate draws provide a separate sensitivity check conditional on the seed-2020 checkpoint, not additional training repeats. Some historical S1 candidate IDs were not retained: all 18 Amazon-Books and six QNN_alpha Yelp2018/ML-1M full-pool coordinates remain UNVERIFIABLE_INPUT. Their candidate IDs and user observations are not inferred from aggregate tables. Learned rerankers trained under DF remain outside the evaluated scope.

Project-owned code is MIT-licensed. Third-party code and references retain their separate terms and attribution. See [LICENSE](LICENSE), [THIRD_PARTY.md](THIRD_PARTY.md), and `licenses/`. No author identity, private server path, or pre-existing Git history is needed to run the package.
