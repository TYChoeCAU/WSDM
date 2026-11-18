# Manuscript tables, figure, and numeric checks

Run these commands from the repository root after installing `requirements.txt`:

```bash
python paper/make_tables.py --results results --out outputs/tables
python paper/make_transfer_fig.py --e3 results/e3 --checks results/checks --out outputs/transfer_fig
python paper/transfer_counts.py --run results/s3
python paper/paper_numbers.py --results results --out outputs/PAPER_NUMBER_CHECK.md
python paper/verify_figure.py --results results --out outputs/figure_value_check.json
```

The first command produces **four** table bodies. Table 1 describes S1/S2/S3, including BCE, BCE+neg., and BPR. Table 2 reports S1 benchmark runs. Table 3 combines fresh U/W/F comparisons with four S1 DF contrasts from `results/followup/DF_s1_paired.csv`; Amazon-Books and S2 have no DF entries. Table 4 contains Base, Pop, NPGC, PRM+PL, Res(p), candidate recall, and oracle. Res(H) and all input subsets remain in the supporting CSVs. There is no fifth manuscript table.

Figure 1 is the manuscript's TikZ pipeline and has no separate data artifact here. The figure command generates **Figure 2** as PDF and PNG. Panels (a,b) read `E3_pop_minus_base_paired.csv`: six series, NFM/MF × {BCE, BCE+neg., BPR}, evaluated under U/W/F on two datasets. Panel (c) reads the three Res(p)−Base effects from `C_paired_repeat_ci.csv`. The reference PDF is `paper/reference/transfer_fig.pdf`.

`verify_figure.py` checks **39 effects with their intervals**: 36 Pop−Base values and three Res(p) values. Paired raw seed differences are recomputed independently of their stored summary. Explicit expected values from the manuscript are checked separately from this complete CSV consistency check. The different repetition counts remain attached to their effects: five training seeds for observed BCE/BPR, three for BCE+neg. The figure's largest 95% interval half-width in panels (a,b) rounds to 2.2 points. Fonts and rendering environments can produce different pixels, while the figure values agree with the CSV evidence.

`paper_numbers.py` verifies Tables 2–4 and numerical claims against `configs/analysis_expected.json`. Printed decimal values use half of the last displayed unit as the rounding tolerance. Integer counts require exact equality. A discrepancy is reported and causes exit 1. It is not resolved by editing evidence. `configs/manuscript_numeric_inventory.json` records the numeric-claim inventory. Figure-specific expected values reside in `configs/figure_expected.json`.

`transfer_counts.py` prints repository-only T/R/E/V/X classifications. It also reports matched/frozen scopes, F-target counts, Recall@20 sensitivity, user-interval agreement, margins of 0.1 and 0.5 points, and headroom-relative margins. X denotes unresolved. V is the post hoc vanish subdivision described in [SETTINGS.md](../SETTINGS.md). The source CSVs retain their original recorded classification fields, while the display uses X for unresolved outcomes.

Headroom recovery averages each seed's `(Pop−Base)/(Oracle−Base)` ratio. It does not divide the aggregate mean differences. Macro summaries first average the specified backbones within each seed, then compute a mean and sample SD across seeds. Neither users nor individual backbone–seed entries are treated as independent training repetitions.

The final S3 PRM+PL fits all use batch 32. The earlier ML-1M U learning-rate selection grids for both NFM and MF-BPR used batch 16384 across four trials. The other ten grids used batch 32. The released configurations preserve this difference; the manuscript's final-fit batch claim is checked against the final fits.

See [../README.md](../README.md) for the manuscript-evidence mapping and repository-only evidence list, [../SETTINGS.md](../SETTINGS.md) for configurations, and [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) for checks actually executed during release preparation. The quick commands only analyze bundled scalar evidence. They do not validate a newly trained model or regenerate excluded per-user arrays.
