"""Regenerates `data/human_abr_summary.json` from `Human_Synaptopathy_ABRdata.csv`.

The human dataset is an unpaired (independent-groups) design: three separate
groups of listeners (`ctrl`, `nexp`, `ma`), not a repeated-measures pre/post
pair like the chinchilla noise-exposure study. This script computes each
group's Wave-I/Wave-V amplitude and LFA (low-frequency pure-tone average,
standing in for a "click threshold" — see the note below) mean/std directly
from the CSV, and writes them into the summary JSON shape `load_abr_dataset`
expects (see `empirical.SpeciesDataFiles`/`empirical._select_group_block`).

Design notes baked into this script's choices — revisit if they change:

- Threshold proxy: the CSV has no true click ABR threshold (unlike
  chinchilla's summary, which has real per-frequency click/tone ABR
  thresholds). Of the three behavioral pure-tone measures available (LFA,
  HFA, EHFA), LFA (low-frequency average) was chosen as the closest analog
  to a broadband click threshold, since clicks are dominated by low/mid
  frequency energy. HFA/EHFA are not used here.
- Single pseudo-frequency: because there's only one Wave-I/Wave-V amplitude
  measurement per subject (no per-frequency breakdown like chinchilla's
  500/1000/2000/4000/8000 Hz series), `frequencies_hz` is just `[0]` and the
  mandatory trailing "tone average" entry duplicates that same single value
  rather than fabricating a second, different one.
- NaN handling: a handful of subjects have the literal string "NaN" in their
  w1/w5 fields (1 in ctrl, 0 in nexp, 3 in ma, out of 55/53/58 respectively).
  Those subjects are excluded from the mean/std for w1/w5 only (`np.nanmean`/
  `np.nanstd`), not dropped from `subjects` or from LFA statistics.
- Per-comparison-group nesting: `thresholds_db_spl`, `high_level_w1_uv`, and
  `high_level_w5_uv` are each nested one level deeper, keyed by "nexp"/"ma",
  because the baseline (`ctrl`) side is shared but the comparison side
  differs depending on which group `--comparison-group`/`comparison_group`
  selects; a flat block (chinchilla's shape) can't represent both.
- Source citation: NOT filled in below (left as a TODO placeholder) since
  this script can't verify which publication the `ctrl`/`nexp`/`ma` grouping
  and CSV come from — fill in `SOURCE_CITATION` once confirmed.

Run from the repository root:
    uv run python scripts/generate_human_abr_summary.py
"""

import csv
import json
import pathlib
from collections import defaultdict

import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "carfac_ephys" / "data"
INPUT_CSV = DATA_DIR / "Human_Synaptopathy_ABRdata.csv"
OUTPUT_JSON = DATA_DIR / "human_abr_summary.json"

# TODO(ben): confirm and fill in the actual citation for this dataset. It is
# NOT Bharadwaj et al. (2022) — that paper's human cohort doesn't use this
# ctrl/nexp/ma independent-groups design (see README.md's "Comparison axis"
# table, which flags the same TODO).
SOURCE_CITATION = "Hari's Human_Synaptopathy_ABRdata.csv (ctrl/nexp/ma design)"

# Which behavioral pure-tone measure stands in for "threshold" (see the
# module docstring above for why LFA was chosen over HFA/EHFA).
THRESHOLD_FIELD = "LFA"

BASELINE_GROUP = "ctrl"
COMPARISON_GROUPS = ("nexp", "ma")


def _group_stat(rows_by_group: dict[str, list[dict[str, str]]], group: str, field: str) -> tuple[float, float, int, int]:
  """Returns (mean, std, n_valid, n_total) for `field` within `group`, skipping NaNs."""
  values = np.array([float(row[field]) for row in rows_by_group[group]])
  n_valid = int(np.sum(~np.isnan(values)))
  return float(np.nanmean(values)), float(np.nanstd(values, ddof=1)), n_valid, len(values)


def main() -> None:
  rows = list(csv.DictReader(INPUT_CSV.open(newline="", encoding="utf-8")))
  rows_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
  for row in rows:
    rows_by_group[row["Group"]].append(row)

  subjects = [row["ID"] for row in rows]

  threshold_stats = {}
  wave1_stats = {}
  wave5_stats = {}
  group_notes = {}

  baseline_threshold = _group_stat(rows_by_group, BASELINE_GROUP, THRESHOLD_FIELD)
  baseline_w1 = _group_stat(rows_by_group, BASELINE_GROUP, "w1")
  baseline_w5 = _group_stat(rows_by_group, BASELINE_GROUP, "w5")

  for group in COMPARISON_GROUPS:
    group_threshold = _group_stat(rows_by_group, group, THRESHOLD_FIELD)
    group_w1 = _group_stat(rows_by_group, group, "w1")
    group_w5 = _group_stat(rows_by_group, group, "w5")

    threshold_stats[group] = {
      "mean_pre": [baseline_threshold[0]],
      "mean_post": [group_threshold[0]],
      "std_pre": [baseline_threshold[1]],
      "std_post": [group_threshold[1]],
    }
    # Trailing entry duplicates the single measurement (see module docstring):
    # there's no separate per-frequency value to report for a real "tone average".
    wave1_stats[group] = {
      "mean_pre": [baseline_w1[0], baseline_w1[0]],
      "mean_post": [group_w1[0], group_w1[0]],
      "std_pre": [baseline_w1[1], baseline_w1[1]],
      "std_post": [group_w1[1], group_w1[1]],
    }
    wave5_stats[group] = {
      "mean_pre": [baseline_w5[0], baseline_w5[0]],
      "mean_post": [group_w5[0], group_w5[0]],
      "std_pre": [baseline_w5[1], baseline_w5[1]],
      "std_post": [group_w5[1], group_w5[1]],
    }
    group_notes[group] = (
      f"{BASELINE_GROUP} n={baseline_threshold[3]} ({THRESHOLD_FIELD}), "
      f"{baseline_w1[2]}/{baseline_w1[3]} valid w1, {baseline_w5[2]}/{baseline_w5[3]} valid w5; "
      f"{group} n={group_threshold[3]} ({THRESHOLD_FIELD}), "
      f"{group_w1[2]}/{group_w1[3]} valid w1, {group_w5[2]}/{group_w5[3]} valid w5"
    )

  summary = {
    "source": SOURCE_CITATION,
    "subjects": subjects,
    "frequencies_hz": [0],
    "thresholds_db_spl": threshold_stats,
    "high_level_w1_uv": wave1_stats,
    "high_level_w5_uv": wave5_stats,
    "notes": {
      "threshold_proxy": (
        f"'{THRESHOLD_FIELD}' (low-frequency pure-tone average) stands in for a click "
        "ABR threshold; the CSV has no true click threshold. HFA/EHFA were not used."
      ),
      "frequencies_hz": (
        "[0] is a placeholder pseudo-frequency, not a real stimulus frequency: there is "
        "only one Wave-I/Wave-V amplitude per subject (no per-frequency series), so the "
        "mandatory trailing 'tone average' entry duplicates the same single value."
      ),
      "nan_handling": (
        "A few subjects have the literal string 'NaN' in w1/w5; those are excluded from "
        "the mean/std for w1/w5 only (not dropped from 'subjects' or from the threshold field)."
      ),
      "group_stats": group_notes,
      "generated_by": "scripts/generate_human_abr_summary.py",
    },
  }

  OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  print(f"Wrote {OUTPUT_JSON} ({len(subjects)} subjects).")


if __name__ == "__main__":
  main()
