# Release reproduction status

The five quick-reproduction commands in the README pass in a new Python 3.10 virtual environment installed from the pinned `requirements.txt`, without system site packages. `pip check` reports no broken requirements. These commands read recorded scalar evidence and use the CPU. No training, model inference, graph query, or feature-cache generation took place during release preparation.

| Command | Verified output | Exit code |
|---|---|---:|
| `paper/make_tables.py` | Tables 1–4, with the four recorded DF cells in Table 3 and seven numeric columns in Table 4 | 0 |
| `paper/make_transfer_fig.py` | Figure 2, with six training series and the frozen residual panel | 0 |
| `paper/transfer_counts.py` | Repository-only T/R/E/V/X counts, including unresolved X | 0 |
| `paper/paper_numbers.py` | 629/629 numerical and table checks pass | 0 |
| `paper/verify_figure.py` | 170/170 value checks pass | 0 |

## Numerical evidence

`configs/analysis_expected.json` contains expected values from the submitted manuscript. Printed metrics use half a unit at their displayed precision. Counts and discrete settings require exact agreement. The checker does not modify result evidence or adjust expectations to fit a result.

`configs/manuscript_numeric_inventory.json` maps 712 numerical occurrences from the abstract, Sections 1–6, captions, pipeline labels, and Tables 1–4. It contains 194 local-claim occurrences, 408 table-value occurrences, 11 table-text occurrences, 66 identifiers, 25 structural descriptions, two mathematical definitions, and six external-reference occurrences. No occurrence remains unmapped. External-reference claims are not independently recalculated from local CSVs and do not count as computational passes.

The E3 evidence contains all 12 required architecture–dataset–training-seed coordinates, or 108 metric rows across U/W/F and Base/Pop/NPGC. E3 uses three training seeds. The four historical training conditions retain five seeds. The optional two E3 seeds were not run. The same-pool accuracy rule agrees in 33/36 conditions, with three exceptions under Yelp2018 F. The 0.25-point margin contains two of those exceptions.

N1 contains 60 completed S3 coordinates. All 54 S1 coordinates remain unavailable because historical candidate-level native NPGC scores were not stored. The release does not infer those missing scores or treat a newly generated candidate list as historical evidence. The correlations describe association, not a causal attribution of NPGC's accuracy gain.

All 65 existing result CSVs remain unchanged. New evidence only normalizes private provenance paths. Numeric cells remain unchanged, as documented in `results/release_provenance.json` and `results/FORMAT.md`.

## Figure values

Figure 1 is the manuscript's TikZ pipeline and has no data-generated artifact here. `verify_figure.py` checks Figure 2's 36 Pop−Base effects in panels (a,b) and three Res(p)−Base effects in panel (c). It independently recomputes all 117 means and interval endpoints from paired seed metrics, checks 36 links to Table 4, checks 16 printed manuscript/reference-figure values, and checks the 2.2-point maximum interval half-width. E3 intervals use three training repeats and two degrees of freedom. Historical intervals use five repeats and four degrees of freedom.

Fonts and rendering environments can produce different pixels, while the figure values agree with the CSV evidence.

The supplied reference PDF and generated PDF/PNG contain no embedded date, time-zone, creator, producer, software, or user-identifying fields. Filesystem modification timestamps are distinct from embedded image metadata.

## Code validation and limits

`tests/test_core_runtime.py` passes all four CPU import, configuration-restoration, and dry-run checks with zero skips in the existing pinned training runtime and patched upstream checkout. The 21 follow-up modules import successfully without training or inference. E3 and N1 have explicit public entrypoints in `scripts/run_followup.py`. The original training, sampling, scoring, and diagnostic operations remain in the extracted implementations, with runtime locations supplied by arguments and manifests. The expired submission-time scheduling gate is not a training hyperparameter and does not restrict a future local rerun.

The full GPU reproduction pipeline was not rerun during packaging. It requires the datasets, selected checkpoints, native scores, and candidate caches described in the README. The release does not bundle those large inputs. A newly reproduced S3 lineage must use its own corresponding MostPop and objective-ablation evidence when E3 aggregates results. The manifest supplies those paths and the existing identity checks remain active.

Table 4's near-tie check confirms a maximum per-seed raw NDCG@20 difference below 0.0003 for the specified Res(p)/Pop cell. Scalar metric agreement alone does not verify that the candidate ordering is identical. Thus the caption's ordering statement is not independently established by the scalar-only release checks.

No result CSV metric, original manuscript, or prior release was changed. No upload or push took place.
