## What is included

This repository contains the small recorded evidence, the code that produced it, and portable entrypoints for new runs. It does not contain datasets, checkpoints, candidate tensors, or per-user prediction arrays. No new training was performed to prepare this release. Numerical reproduction from bundled CSVs was tested in a fresh virtual environment. Full training entrypoints were checked through imports, configuration restoration, and data preparation checks, but the complete training suite was not rerun during packaging.

| Manuscript element | Evidence | Reproduction |
|---|---|---|
| Table 1: study design | `results/studies.csv` | `paper/make_tables.py` |
| Table 2: S1 benchmark | `results/s1/benchmark_seed_metrics.csv`, `s1_candidate_recall.csv` | `paper/make_tables.py`; `scripts/run_s1.py benchmark` |
| Table 3: S1 fresh pools/catalog, S2 frozen MF | `results/s1/fresh_seed_metrics.csv`, `results/s2/seed_metrics.csv` | `paper/make_tables.py`; S1 `fresh`, S2 `evaluate` |
| Table 4: S3 matched methods | `results/s3/homogeneous_method_summary.csv` | `paper/make_tables.py`; `scripts/analyze_s3.py` |
| Table 5: transfer counts and Res(p) fit/evaluation | `results/s3/transfer_classification.csv`, `results/checks/C_paired_repeat_ci.csv` | `paper/transfer_counts.py`, `paper/make_tables.py` |
| Figure 1 | `results/s3/repeat_intervals.csv`, `results/checks/C_paired_repeat_ci.csv` | `paper/make_transfer_fig.py` |
| Sections 4.1–4.3 numerical claims | S1–S3 scalar evidence, MostPop, identity, Spearman and sensitivity CSVs | `paper/paper_numbers.py` |

See [results/FORMAT.md](results/FORMAT.md) for keys, units, uncertainty levels, and provenance limitations. `artifact:` and `historical/` references are logical provenance identifiers, not required local paths.

## Quick reproduction without training

Run from this repository's root. Python 3.10 is supported by the pinned analysis environment. No GPU or upstream code is required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR="$PWD/.mplconfig"
python paper/make_tables.py --results results --out outputs/tables
python paper/make_transfer_fig.py --run results/s3 --checks results/checks --out outputs/transfer_fig
python paper/transfer_counts.py --run results/s3
python paper/paper_numbers.py --results results --out outputs/PAPER_NUMBER_CHECK.md
```

The commands take seconds after installation on a typical CPU. Allow a few minutes for dependency installation. Table bodies use `booktabs` and the manuscript's `\Score` macro. The number checker writes both Markdown and JSON reports, then exits 1 if a manuscript discrepancy remains.

The recorded evidence passes **660/660 numerical checks**, including every numeric cell in Tables 2–5 at manuscript precision. Expected values from the manuscript are stored in `configs/analysis_expected.json`. The result CSVs remain unchanged. See `paper/REPRODUCTION_STATUS.md` for the validation scope.

Check the 15 plotted effects and their confidence intervals directly from the CSVs:

```bash
python paper/verify_figure.py --results results --out outputs/figure_value_check.json
```

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
* **S3** uses NFM and MF-BPR on Yelp2018 and ML-1M. Seeds 2020–2024 are training repeats on one fixed split and fixed candidate draw. U/W request 100 unique negatives per relevant item. F scores eligible catalog items. Candidate IDs, labels, validity masks, and full positive-count denominators are preserved for aligned evaluation. Top-50 ties use item ID. R0 training batches are 4096 for backbones, 256 for residual modules, and 32 for PRM+PL. The 20 backbone settings and 60 protocol-specific applied bundles are under `configs/s3/`.
* **NPGC** uses k=2, except k=5 on Amazon-Books. The native co-occurrence normalization and self-neighbor treatment differ from residual graph features and are not substituted for each other. `results/checks/A_npgc_equivalence.csv` records the two actual local-versus-official checks with tolerance 1e-3. Those two checks do not certify every possible coordinate.
* **Residuals** use inputs `[s, log1p_item_degree, log1p_q, n_eff, h, h_minus_s, variance]`, a 7→32→16→1 MLP, and a zero-initialized output layer. The prediction is `s + t * f(z)`. Training-set mean/std define `z`. The eight S/P/R/PR/H/RH/PH/PRH masks zero inactive coordinates after standardization. The scale `t` is the training candidate score standard deviation with the recorded floor. AdamW uses lr=1e-3, weight_decay=1e-4, clip norm 5, at most 50 epochs, minimum 5 epochs, patience 8, and min_delta=1e-4. The group-averaged score-scaled pairwise loss is preserved. Identity-inclusive and trained-only checkpoint policies remain separate. Frozen evaluation retains the source model, scaler, and scale.
* **PRM+PL** uses the recorded selected learning rates from the original `{1e-4, 3e-4}` validation comparison. Full runs inherit the coordinate's final configuration rather than perform a new test-informed search. Batch size is 32 and the maximum is 100 epochs. S3 selection uses validation NDCG@20. Backbone and reranker selection are separate.

The actual JSON settings and executable paths take precedence over these summaries. No new split, feature definition, loss, or hyperparameter budget was introduced for this release.

## Statistics and interpretation

The primary margin is **0.25 NDCG@20 points**, equivalent to 0.0025 on the 0–1 metric scale. Paired training-repeat contrasts use Student-t 95% effect intervals and 90% equivalence intervals. A transfer requires both 95% intervals to be beyond the margin with the same sign. A reversal requires the 95% intervals to have opposite signs beyond the margin. Equivalence requires both 90% intervals to lie inside the margin. Other comparisons are unresolved. The displayed **vanish** category requires a source 95% interval beyond the margin and a target 90% interval inside the margin. It subdivides unresolved outcomes after inspecting outcomes and is not a preregistered independent test.

The 16 primary TOST hypotheses test protocol changes in two method contrasts, Pop−Base and PRM+PL−Res(p), over four dataset/backbone cells and two transitions, F−U and F−W. Holm adjustment applies to this family. A same-sign transfer does not imply equality of effect sizes. The 136 comparisons comprise 96 matched plus 40 frozen comparisons after removing duplicate fixed-rule frozen contrasts. Seed-wise paired user bootstrap uses 2,000 resamples and is reported separately from variation across training repeats. Five seeds are not 2,000 independent repetitions. `configs/s3/analysis_plan.json` records margins, tie handling, identity policy, and the fact that this analysis plan followed earlier outcome inspection.

## Scope, limitations, and licensing

DNR model code, DNR results, R1 large-batch results, P3, and incomplete NFM-BPR experiments are excluded. The historical directory name `baselines/dnr` contains only shared data/backbone adapters required for candidate generation, not the DNR method. Completed S1 E4 catalog evidence is included. S3 does not establish uncertainty over alternative data splits or candidate draws. Some historical S1 candidate IDs were not retained; their identity limitations remain explicit in the released CSVs. The package does not infer missing user observations from aggregate tables.

Project-owned code is MIT-licensed. Third-party code and references retain their separate terms and attribution. See [LICENSE](LICENSE), [THIRD_PARTY.md](THIRD_PARTY.md), and `licenses/`. No author identity, private server path, or pre-existing Git history is needed to run the package.
