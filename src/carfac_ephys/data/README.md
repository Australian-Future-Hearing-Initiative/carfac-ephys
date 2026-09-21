# Human source data

The committed `human_abr_summary.json` contains group statistics and counts only.
Individual-level human ABR and MEMR CSVs are private source files and must not be
committed. Both `Human_Synaptopathy_ABRdata.csv` and
`Human_Synaptopathy_MEMRdata.csv` are ignored at any directory depth.
Dataset citation and redistribution permission remain unconfirmed.

To regenerate with an authorized local copy, run from the repository root:

```sh
uv run python scripts/generate_human_abr_summary.py --csv /private/path/Human_Synaptopathy_ABRdata.csv
```

Without `--csv`, the generator looks for the private ABR CSV in this directory.
The generator never includes participant identifiers in the summary. Preserve
private sources outside the checkout before switching to older branches.

Generator tests use synthetic observations. Only the real-source drift test is
skipped when the default private CSV is absent; committed-summary checks always run.
