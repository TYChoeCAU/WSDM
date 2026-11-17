# Release reproduction status

The CSV-only commands below were executed in a new Python 3.10 virtual environment installed from `requirements.txt`, without system site packages. They require no GPU, model checkpoints or private experiment modules. `pip check` passed.

| README command | Result | Exit code |
| --- | --- | --- |
| `paper/make_tables.py` | Tables generated. Tables 2–4 remain byte-identical to the previous release outputs. | 0 |
| `paper/make_transfer_fig.py` | Four-series figure generated from the completed CSV evidence. | 0 |
| `paper/transfer_counts.py` | Counts generated with unresolved category `X`. | 0 |
| `paper/paper_numbers.py` | 720/721 checks pass. One recorded manuscript-setting discrepancy remains. | 1 |
| `paper/verify_figure.py` | 125/125 value checks pass. | 0 |

## Numerical claims

Expected values from the submitted manuscript are stored in `configs/analysis_expected.json`. All checked table values and numerical performance claims pass at their displayed precision. Counts and discrete settings require exact agreement. The evidence and expected settings have not been changed to conceal the following discrepancy.

The manuscript describes PRM+PL learning-rate selection with batch size 32. Two ML-1M/U selection grids, for NFM and MF-BPR, record batch size 16384 across four trials. The other ten selection grids use 32. All 60 formal R0 PRM+PL fits separately use batch size 32. This makes `setting.prm_selection_batch` fail, so `paper_numbers.py` intentionally returns exit code 1 and lists the discrepancy in its report.

`configs/manuscript_numeric_inventory.json` classifies 825 numerical tokens from the abstract, main text, captions and Tables 2–5, with no unmapped tokens. It distinguishes 222 local-claim tokens, 524 table-value tokens, 51 identifiers, 17 structural descriptions, three mathematical or protocol definitions, seven external-reference tokens and one qualitative approximation. External-reference claims are not independently recalculated from the local evidence. The phrase “a third” remains a disclosed approximation, with its exact evidence ratios reported separately. Neither category is counted as a computational PASS.

## Figure values

`verify_figure.py` checks all 24 Pop−Base points in panels (a) and (b), plus the three Res(p)−Base points in panel (c). It checks 81 means and interval endpoints against seed-level evidence, 36 connections to Table 4, and eight printed manuscript expectations, for 125 checks in total.

Fonts and rendering environments can produce different pixels, while the figure values agree with the CSV evidence.

## Code validation and preservation

`tests/test_core_runtime.py` passed all four tests with zero skips in a separate existing runtime containing the pinned upstream implementation plus its published patches. The new follow-up entry point passed seven help checks in the new analysis environment. All 17 follow-up modules imported in the upstream-enabled runtime. Static checks found 19 selected numerical implementations structurally identical to their original functions, with all four recorded E training profiles preserved. These static and import checks do not establish end-to-end numerical reproduction of the extracted GPU pipeline.

The 73 existing core source files and 40 existing result CSVs remain byte-identical. The 26 added evidence files preserve their numerical values. Only provenance paths are anonymized, as detailed in `results/FORMAT.md`. The original release copy and manuscript remain unchanged. The public source and generated PDF/PNG files passed identity and embedded-metadata checks.

No new model fitting, catalog inference, graph query or feature-cache generation was performed during this update. Full follow-up execution requires the data, selected checkpoints and candidate artifacts described in the README and explicit local manifest. Missing historical identity records remain marked as unavailable. Import checks and CSV reproduction do not certify those missing records.
