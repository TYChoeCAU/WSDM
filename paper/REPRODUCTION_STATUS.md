# Release reproduction status

The CSV-only commands were executed in a newly created Python 3.10 virtual environment installed from `requirements.txt`. No site packages, private experiment modules, model checkpoints, or GPU were used. `pip check`, table generation, figure generation, and the supplied transfer-count command passed. The paper commands complete in seconds after installation.

## Numerical claims

`paper_numbers.py` reports **660 PASS / 0 MISMATCH**, with exit code 0. Every numeric cell in Tables 2–5 agrees at manuscript precision. Expected values from the submitted manuscript are stored in `configs/analysis_expected.json`. Raw result CSVs remain unchanged.

## Figure values

`verify_figure.py` reads the 12 Pop−Base effects and intervals in panels (a) and (b), plus the three Res(p)−Base effects and intervals in panel (c), directly from the CSVs and compares them with manuscript expectations.

```bash
python paper/verify_figure.py --results results --out outputs/figure_value_check.json
```

Fonts and rendering environments can produce different pixels, while the figure values agree with the CSV evidence.

## Full pipeline validation

The fixed configurations and executable imports were checked with a separate copy of the pinned official upstream commit plus the two published patches. S1 restored all 54 actual configurations. S2 enumerated six final repeats. S3 validated 720 matched method coordinates, 20 backbone settings, and its R0 batch constraints. CPU data-preparation checks compare recorded split and candidate-pool digests. Numerical functions were compared structurally against their original implementation before extraction.

These checks do not establish that every complete GPU run has been reproduced in the extracted repository. No new model fitting, catalog model inference, or graph query was performed during release preparation. The full suite is deliberately left to the explicit training commands, separately from the tested CSV-only reproduction path.
