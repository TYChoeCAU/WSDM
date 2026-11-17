# Manuscript reproduction

Run these commands from the repository root:

```bash
python paper/make_tables.py --results results --out outputs/tables
python paper/make_transfer_fig.py --followup results/followup --checks results/checks --out outputs/transfer_fig
python paper/transfer_counts.py --run results/s3
python paper/paper_numbers.py --results results --out outputs/PAPER_NUMBER_CHECK.md
python paper/verify_figure.py --results results --out outputs/figure_value_check.json
```

These commands use only the bundled CSVs. They do not load a model, train, or use a GPU. The figure generator reads the saved effect estimates and confidence intervals from CSVs. It creates the output directory and omits identifying PDF metadata. `transfer_counts.py` retains the recorded transfer-counting rules and labels unresolved outcomes X. `make_tables.py` produces all five table bodies and `table_values.json`; table bodies require `booktabs` and the manuscript's `\Score` macro. Table 2 is recomputed from per-seed metrics and coverage. Table 3 uses paired per-seed differences and Student-t intervals. Table 4 uses saved five-seed method summaries. Table 5 contains only the five T/R/E/V/X comparison rows. Frozen Res(p) effects remain in Figure 1(c) and the evidence CSVs.

`paper_numbers.py` compares body claims and every numeric cell in Tables 2–5 against the submitted manuscript. `configs/analysis_expected.json` contains expected values from the manuscript; it is never used to generate a table or modify evidence. Tolerances are half a unit at each printed precision, and integer counts must match exactly. The command exits 1 if any mismatch remains, after writing the complete Markdown and JSON reports.

The checker covers retained claims plus the architecture × training comparison, S1 full-pool ordering, S1/S3 decoupled retrieval, all-pairs outcomes, and candidate-draw sensitivity. The maximum relative S1 NPGC gain on fixed Top-50 lists is checked against the submitted manuscript value of 10.9%. See `REPRODUCTION_STATUS.md` for actual check counts. Numerical result values are unchanged; only private provenance paths are anonymized.

The current check flags one setting discrepancy and exits 1: PRM+PL final R0 fits use batch 32, while the two recorded ML-1M U learning-rate selection grids used batch 16384. The manuscript describes batch 32 without this distinction. Result evidence and expected manuscript values remain separate. External original-paper claims are listed as not recalculated from local CSVs and are not counted as computational passes.

The figure has 24 Pop−Base effects from four architecture/training series across two datasets and three protocols, plus three Res(p)−Base effects. Verify their estimates and confidence intervals against the evidence and manuscript expectations:

```bash
python paper/verify_figure.py --results results --out outputs/figure_value_check.json
```

Fonts and rendering environments can produce different pixels, while the figure values agree with the CSV evidence.

The submitted manuscript describes training-dependent popularity effects, superiority to NPGC on the recorded whole pools, and surviving popularity gains under decoupled retrieval. These statements apply to the evaluated settings. The objective/exposure swap changes several training factors together, and the five candidate draws are conditional on one frozen checkpoint.
