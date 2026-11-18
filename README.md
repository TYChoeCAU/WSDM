# What Does a Re-ranking Gain on Sampled Pools Measure? A Case Study with a Popularity Control

Anonymous review artifact for the submitted manuscript. See [SETTINGS.md](SETTINGS.md) for every recorded hyperparameter, seed budget, selection grid, and configuration source. [results/FORMAT.md](results/FORMAT.md) defines evidence keys, units, and repetition units.

The study asks three questions:

- **RQ1 — The popularity control:** does NPGC improve over a simple popularity ordering? Pop exceeds NPGC in 54/54 benchmark Top-50 runs and 30/30 runs with retained whole sampled pools.
- **RQ2 — The role of the backbone:** how does the control depend on training? Two architectures, NFM and MF, are tested under observed-rating BCE, BCE with added unobserved negatives, and BPR. Under U, the sign follows exposure to unobserved negatives in these tested conditions. Changing BCE to BPR also changes training-row composition. This comparison does not isolate a causal loss-function effect.
- **RQ3 — Catalog candidates:** does a sampled-pool conclusion persist under F or decoupled retrieval (DF)? Effect sizes change, coverage can limit headroom, and a frozen Res(p) can reverse its sampled lead. The released transfer classifications distinguish practical transfer, reversal, equivalence, vanish, and unresolved outcomes.

## What is included

The repository contains recorded scalar evidence, configurations, core implementations, and portable execution entrypoints. Datasets, checkpoints, full candidate tensors, and per-user prediction arrays are excluded. The CSV-only commands below require neither those large inputs nor a GPU. Training and inference entrypoints need prepared data, selected checkpoints, and the artifacts specified by their arguments. Release validation checks those paths through imports and core tests, without rerunning experiments. See [paper/REPRODUCTION_STATUS.md](paper/REPRODUCTION_STATUS.md) for the actual validation results.

The manuscript has **Tables 1–4**. Its Figure 1 is a TikZ pipeline inside the manuscript, so it has no data-generated repository artifact. **Figure 2** is the numerical figure generated here.

| Manuscript element | Evidence files relative to `results/` | CSV-only reproduction command |
|---|---|---|
| Table 1, study design (§3) | `studies.csv` | `python paper/make_tables.py --results results --out outputs/tables` |
| Table 2, benchmark Top-50 (§4.1) | `s1/benchmark_seed_metrics.csv`, `s1/s1_candidate_recall.csv` | `python paper/make_tables.py --results results --out outputs/tables` |
| Table 3, fresh U/W/F and DF (§4.1–4.2) | `s1/fresh_seed_metrics.csv`, `s2/seed_metrics.csv`, `followup/DF_s1_paired.csv` | `python paper/make_tables.py --results results --out outputs/tables` |
| Table 4, matched methods and coverage (§4.3) | `s3/homogeneous_method_summary.csv`, `s3/extracted_seed_results.csv` | `python paper/make_tables.py --results results --out outputs/tables` |
| Figure 2(a,b), six training conditions | `e3/E3_pop_minus_base_paired.csv`, `e3/E3_all_conditions_seed_results.csv` | `python paper/make_transfer_fig.py --e3 results/e3 --checks results/checks --out outputs/transfer_fig` |
| Figure 2(c), frozen Res(p) | `checks/C_paired_repeat_ci.csv`, `checks/C_paired_seed_differences.csv` | Same figure command; `python paper/verify_figure.py --results results --out outputs/figure_value_check.json` |
| Abstract, §1 and §4.1, popularity and whole-pool claims | `s1/benchmark_seed_metrics.csv`, `followup/S1_fullpool_*.csv`, `followup/DF_s1_*.csv`, `checks/D_mostpop*.csv` | `python paper/paper_numbers.py --results results --out outputs/PAPER_NUMBER_CHECK.md` |
| §3 settings, §4.2 training and score diagnostics | `e3/E3_accuracy_rule_36.csv`, `e3/E3_training_3x2_table.csv`, `e3/N1_npgc_popularity_summary.csv`; recorded `configs/` | Same number-check command |
| §4.3 coverage, transfer, frozen effects; §5 candidate sensitivity | `s3/transfer_classification.csv`, `s3/extracted_seed_results.csv`, `followup/DF_frozen_learned.csv`, `followup/P4_transfer.csv`, `checks/C_*.csv` | Same number-check command; `python paper/transfer_counts.py --run results/s3` |
| Figure 2 caption, Table 4 caption and remaining numerical claims | Raw seed results, paired intervals, `configs/analysis_expected.json`, `configs/figure_expected.json` | Number-check command and figure-value command above |

The checks compare evidence with expected values from the manuscript. A mismatch produces a nonzero exit code rather than changing evidence. Manuscript tables use a 100-point metric scale. Paired differences, relative percentages, and uncertainty units remain separate.

## Results retained only in the repository

These supporting results are not additional manuscript tables. A numeric-summary command reads bundled evidence. Regenerating user-level statistics requires the omitted predictions and caches.

| Supporting evidence | Files relative to `results/` | Inspect or reproduce |
|---|---|---|
| Res(H), score-only control, all eight input subsets | `s3/extracted_seed_results.csv`, `s3/homogeneous_method_summary.csv`; masks in `src/studies/s3/cross_common.py` | Inspect the CSVs; regenerate from local predictions with `python scripts/analyze_s3.py --run runs/s3` |
| Complete T/R/E/V/X counts, including unresolved X | `s3/transfer_classification.csv` | `python paper/transfer_counts.py --run results/s3` |
| Primary 16-hypothesis TOST family | `s3/tost_holm_primary.csv` (`registered_primary_family=True`) | Inspect the CSV; `python paper/paper_numbers.py --results results --out outputs/PAPER_NUMBER_CHECK.md` |
| Recall@20 | `s1/benchmark_seed_metrics.csv`, `s2/seed_metrics.csv`, `s3/extracted_seed_results.csv`, `e3/E3_summary.csv` | Inspect metric columns; `python paper/transfer_counts.py --run results/s3` prints matched Recall decisions |
| Agreement with seed-wise user intervals | `s3/transfer_classification.csv`, `s3/user_bootstrap.csv` | `python paper/transfer_counts.py --run results/s3` |
| Expanded 15-pair comparison (P0) | `followup/P0_extended_counts.csv`, `P0_contrast_details.csv`, `P0_all_pairs_reversals.csv`, `P0_sign_flips.csv` | Number-check command; optional scalar reaggregation: `python scripts/run_followup.py p0 --evidence-root results --out runs/followup` |
| Initial identity-scorer selection | `checks/C_identity_selection.csv`, `s3/extracted_seed_results.csv` | Number-check command; regenerate with `python scripts/run_diagnostics.py identity --s3-run runs/s3 --out runs/checks` |
| Candidate-draw variance (P4) | `followup/P4_seed_results.csv`, `P4_draw_variance.csv`, `P4_transfer.csv` | Number-check command; inference regeneration requires the frozen checkpoint, as documented below |
| Margins 0.1 and 0.5 points; headroom-relative sensitivity | `s3/transfer_classification.csv`, `s3/homogeneous_method_summary.csv` | `python paper/transfer_counts.py --run results/s3` |
| Label-free NPGC/popularity diagnostics (N1) | `e3/N1_npgc_popularity_seed.csv`, `N1_npgc_popularity_summary.csv`, `N1_coordinate_inventory.csv` | Number-check command; `n1` reads locally stored native scores without training or inference |

The whole-pool analysis covers 30 measured S1 coordinates. All 18 Amazon-Books coordinates and six QNN_alpha coordinates lack retained historical pre-Top50 pool inputs. N1 has a different limitation: Top-50 candidates exist for all 54 S1 coordinates, but stored candidate-level native NPGC scores do not, so those N1 rows are unavailable. S3 N1 covers 60/60 coordinates. Its within-list correlations describe score/popularity association, not the cause of a performance gain.

## Quick reproduction without training

Run from the repository root with Python 3.10. The pinned analysis environment needs neither upstream model code nor a GPU.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR="$PWD/.mplconfig"
python paper/make_tables.py --results results --out outputs/tables
python paper/make_transfer_fig.py --e3 results/e3 --checks results/checks --out outputs/transfer_fig
python paper/transfer_counts.py --run results/s3
python paper/paper_numbers.py --results results --out outputs/PAPER_NUMBER_CHECK.md
python paper/verify_figure.py --results results --out outputs/figure_value_check.json
```

These commands take seconds after dependency installation. Table bodies use `booktabs` and the manuscript's `\Score` macro. The number checker writes Markdown and JSON reports. Figure validation checks **39 effects and their intervals**: 36 Pop−Base points for NFM/MF × three training conditions × U/W/F × two datasets, plus three frozen Res(p)−Base effects. It recomputes numeric evidence rather than comparing pixels.

Expected values from the manuscript are in `configs/analysis_expected.json` and `configs/figure_expected.json`. The former covers Tables 2–4 and the numerical claims. The figure checker validates all plotted values against raw seed records, with explicit manuscript values checked separately. Fonts and rendering environments can produce different pixels, while the figure values agree with the CSV evidence.

See [paper/README.md](paper/README.md) for output interpretation. Private provenance strings are normalized as recorded in `results/release_provenance.json`. `artifact:` and historical run references identify omitted evidence, not files that must be present under this repository.

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

## Follow-up execution paths

`scripts/run_followup.py` exposes the code that generated the additional evidence. This release preparation checked imports and command help only. It did not rerun these training or inference stages. The bundled CSVs are sufficient for the quick paper commands, but are not substitutes for data, selected checkpoints, full candidate IDs, features, or predictions.

| Task | Execution | Required inputs |
|---|---|---|
| `e1`, `e2` | Train NFM-BPR or MF-BCE, evaluate Base/Pop/NPGC, or aggregate | Prepared data, S3 fixed split/pools, `configs/followup/declaration.json` |
| `e3` | Train BCE with one added unobserved negative per positive, evaluate Base/Pop/NPGC, or aggregate all six conditions | Prepared data, S3 fixed split/pools, original backbone profiles, `configs/e3/fixed_settings.json`; three training seeds |
| `n1` | Read stored Top-50 scores to measure NPGC/popularity association | Explicit candidate caches and native-score predictions in a local manifest; no checkpoint loading, new inference, or labels |
| `df-s3` | MF-BPR retrieval with NFM scoring, then optional frozen source rerankers | S3 selected backbone checkpoints, MF F candidates, U/W reranker checkpoints, source scaler and scale |
| `df-s1` | Six S1 scorers on same-split S2 MF candidates | Explicit S1/S2 checkpoint, split, candidate, neighbor and lineage entries in a local manifest |
| `fullpool` | Score stored pre-Top50 pools with Base, NPGC, Pop | Real historical or newly retained full pool IDs and scores, labels, full truth, selected backbone and native neighbor table |
| `p0` | Expanded and all-pairs transfer aggregation | Per-repeat S3 numeric result CSVs |
| `p4` | Five candidate RNG draws with a frozen seed-2020 checkpoint | S3 fixed split, selected checkpoint, original sampler inputs; no new model fitting |

`--help` lists stages and coordinate filters. `--stage aggregate` reads completed outputs. `e1`, `e2`, and `e3` with `--stage train` or `all` perform training, while DF, full-pool, and candidate-draw tasks perform inference. `n1` performs only CPU diagnostics on stored scores. Run those commands only when the required large inputs have been prepared.

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

For the added exposure condition and stored-score diagnostic:

```bash
# This E3 command trains one coordinate, then evaluates it. Large inputs are required.
python scripts/run_followup.py e3 --stage all --architecture MF --dataset ml-1m --seed 2020 --data-root "$DATA_ROOT" --s3-run runs/s3 --out runs/followup --device cuda:0
# After all intended coordinates exist, aggregate their saved metrics.
python scripts/run_followup.py e3 --stage aggregate --s3-run runs/s3 --manifest local_followup_manifest.json --out runs/followup
# This command reads only paths supplied in a completed local manifest.
python scripts/run_followup.py n1 --manifest local_followup_manifest.json --s3-run runs/s3 --out runs/followup
```

For E3 aggregation, `manifest.inputs.mostpop` and `manifest.inputs.objective_seed_results` must refer to the MostPop and earlier objective-condition evidence from the same regenerated S3 master. Bundled historical CSVs cannot stand in for a new run with a different master digest; the lineage check intentionally rejects that mismatch.

E3 uses architectures `MF` or `NFM`, both datasets, and seeds 2020–2022. Its summary retains five repeats for the pre-existing BCE/BPR conditions and three for BCE+neg. It does not pool those observations as equal-sized uncertainty samples. N1 accepts explicit stored-score/cache mappings from the manifest and marks unavailable inputs instead of reconstructing or inferring missing scores. The supplied manifest is an input template, not a bundle of working checkpoints.

## Statistics and interpretation

The primary margin is **0.25 NDCG@20 points**, equivalent to 0.0025 on the 0–1 metric scale. Paired training-repeat contrasts use Student-t 95% effect intervals and 90% equivalence intervals. A transfer requires both 95% intervals to be beyond the margin with the same sign. A reversal requires the 95% intervals to have opposite signs beyond the margin. Equivalence requires both 90% intervals to lie inside the margin. Other comparisons are unresolved. The displayed **vanish** category requires a source 95% interval beyond the margin and a target 90% interval inside the margin. It subdivides unresolved outcomes after inspecting outcomes and is not a preregistered independent test.

The 16 primary TOST hypotheses test protocol changes in two method contrasts, Pop−Base and PRM+PL−Res(p), over four dataset/backbone cells and two transitions, F−U and F−W. Holm adjustment applies to this family. A same-sign transfer does not imply equality of effect sizes. The 136 comparisons comprise 96 matched plus 40 frozen comparisons after removing duplicate fixed-rule frozen contrasts. Seed-wise paired user bootstrap uses 2,000 resamples and is reported separately from variation across training repeats. Five seeds are not 2,000 independent repetitions. `configs/s3/analysis_plan.json` records margins, tie handling, identity policy, and the fact that this analysis plan followed earlier outcome inspection.

## Scope, limitations, and licensing

DNR model code, DNR results, and R1 large-batch results are excluded. The historical directory name `baselines/dnr` contains only shared data/backbone adapters required for candidate generation, not the DNR method. Completed S1 E4 catalog evidence and the requested follow-up results are included. S3 does not establish uncertainty over alternative data splits. BCE+neg. has three training repeats, while the earlier BCE/BPR conditions have five. Five fresh candidate draws provide a separate sensitivity check conditional on the seed-2020 checkpoint, not additional training repeats. Some historical S1 candidate IDs were not retained: all 18 Amazon-Books and six QNN_alpha Yelp2018/ML-1M full-pool coordinates remain UNVERIFIABLE_INPUT. Their candidate IDs and user observations are not inferred from aggregate tables. Learned rerankers trained under DF remain outside the evaluated scope.

Project-owned code is MIT-licensed. Third-party code and references retain their separate terms and attribution. See [LICENSE](LICENSE), [THIRD_PARTY.md](THIRD_PARTY.md), and `licenses/`. No author identity, private server path, or pre-existing Git history is needed to run the package.
