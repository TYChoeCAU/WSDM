# Manuscript reproduction

Run these commands from the repository root:

```bash
python paper/make_tables.py --results results --out outputs/tables
python paper/make_transfer_fig.py --run results/s3 --checks results/checks --out outputs/transfer_fig
python paper/transfer_counts.py --run results/s3
python paper/paper_numbers.py --results results --out outputs/PAPER_NUMBER_CHECK.md
```

These commands use only the bundled CSVs. They do not load a model, train, or use a GPU. The figure generator reads the saved effect estimates and confidence intervals from CSVs. It creates the output directory and omits identifying PDF metadata. `transfer_counts.py` retains the recorded transfer-counting rules. `make_tables.py` produces all five table bodies and `table_values.json`; table bodies require `booktabs` and the manuscript's `\Score` macro. Table 2 is recomputed from per-seed metrics and coverage. Table 3 uses paired per-seed differences and Student-t intervals. Table 4 uses saved five-seed method summaries. Table 5 uses the same classification functions as the supplied counting script and the saved Res(p) fit/evaluation summaries.

`paper_numbers.py` compares body claims and every numeric cell in Tables 2–5 against the submitted manuscript. `configs/analysis_expected.json` contains expected values from the manuscript; it is never used to generate a table or modify evidence. Tolerances are half a unit at each printed precision, and integer counts must match exactly. The command exits 1 if any mismatch remains, after writing the complete Markdown and JSON reports.

The archived evidence passes **660/660 checks**, including all numeric cells of Tables 2–5 and all Table 5 counts. The maximum relative S1 NPGC gain is checked against the submitted manuscript value of 10.9%. Result CSVs are unchanged.

Verify all 15 plotted effects and their confidence intervals from the CSVs against manuscript expectations:

```bash
python paper/verify_figure.py --results results --out outputs/figure_value_check.json
```

Fonts and rendering environments can produce different pixels, while the figure values agree with the CSV evidence.
