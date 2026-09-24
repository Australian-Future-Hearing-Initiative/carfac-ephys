"""Regenerates `data/human_abr_summary.json` from the private human ABR source CSV.

The source CSV holds individual-level recordings and is never committed. Keep
   your copy in `data/private/`, which Git ignores, or point `--csv` anywhere else.
   Only the aggregate summary in `src/carfac_ephys/data/` is tracked.

   Run from the repository root:
       uv run python scripts/generate_human_abr_summary.py
       uv run python scripts/generate_human_abr_summary.py --csv /path/to/source.csv
"""

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "carfac_ephys" / "data"
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "private" / "Human_Synaptopathy_ABRdata.csv"
DEFAULT_OUTPUT_JSON = DATA_DIR / "human_abr_summary.json"

SOURCE_CITATION = "INSERT CITATION HERE for the human ABR dataset"

GROUP_COLUMN = "Group"
BASELINE_GROUP = "ctrl"
COMPARISON_GROUPS = ("nexp", "ma")

# Measure name -> (CSV column, units, description).
MEASURES: dict[str, tuple[str, str, str]] = {
  "wave1_uv": ("w1", "uV", "High-level click ABR Wave-I amplitude."),
  "wave5_uv": ("w5", "uV", "High-level click ABR Wave-V amplitude."),
  "audiometric_lfa_db_hl": (
    "LFA",
    "dB HL",
    "Behavioural low-frequency pure-tone average. Listener context only: not an "
    "ABR threshold, and not comparable with a simulated threshold in dB SPL.",
  ),
  "audiometric_hfa_db_hl": (
    "HFA",
    "dB HL",
    "Behavioural high-frequency pure-tone average. Listener context only: not an "
    "ABR threshold, and not comparable with a simulated threshold in dB SPL.",
  ),
  "audiometric_ehfa_db_hl": (
    "EHFA",
    "dB HL",
    "Behavioural extended-high-frequency pure-tone average. Listener context "
    "only: not an ABR threshold, and not comparable with a simulated threshold "
    "in dB SPL.",
  ),
}


# A sample standard deviation is undefined below two observations.
MIN_VALID_OBSERVATIONS = 2


def _measure_stats(values: Sequence[float], measure: str, group: str) -> dict[str, float | int]:
  """Returns mean, sample std and valid n for one measure, rejecting undefined statistics.

  Subjects carrying the literal string "NaN" are excluded from that measure
  only; they still count towards the group's own `n`. No identifiers are
  emitted. An infinite observation, or too few valid observations for a sample
  standard deviation, is an error rather than a non-finite value written into
  the JSON.
  """
  array = np.asarray(values, dtype=float)
  if bool(np.isinf(array).any()):
    raise ValueError(f"{group}/{measure}: infinite observation in the source data.")
  valid = array[~np.isnan(array)]
  if valid.size < MIN_VALID_OBSERVATIONS:
    raise ValueError(
      f"{group}/{measure}: {valid.size} valid observation(s); at least "
      f"{MIN_VALID_OBSERVATIONS} are needed for a sample standard deviation."
    )
  mean = float(np.mean(valid))
  std = float(np.std(valid, ddof=1))
  if not math.isfinite(mean) or not math.isfinite(std):
    raise ValueError(f"{group}/{measure}: computed non-finite statistics (mean={mean}, std={std}).")
  return {"mean": mean, "std": std, "n": int(valid.size)}


def build_summary(csv_path: pathlib.Path = DEFAULT_INPUT_CSV) -> dict:
  """Builds the summary structure from the per-subject CSV."""
  with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
  if not rows:
    raise ValueError(f"No rows found in {csv_path}.")

  expected_groups = (BASELINE_GROUP, *COMPARISON_GROUPS)
  rows_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
  for line_number, row in enumerate(rows, start=2):  # line 1 is the header
    label = row[GROUP_COLUMN].strip()
    if label not in expected_groups:
      raise ValueError(
        f"{csv_path} line {line_number}: unexpected group '{label}'; "
        f"expected one of {list(expected_groups)}."
      )
    rows_by_group[label].append(row)
  missing = [group for group in expected_groups if group not in rows_by_group]
  if missing:
    raise ValueError(f"{csv_path} is missing group(s) {missing}; found {sorted(rows_by_group)}.")

  groups: dict[str, dict] = {}
  for group in expected_groups:
    group_rows = rows_by_group[group]
    stats: dict[str, object] = {"n": len(group_rows)}
    for measure, (column, _units, _description) in MEASURES.items():
      stats[measure] = _measure_stats(
        [float(row[column]) for row in group_rows], measure=measure, group=group
      )
    groups[group] = stats

  return {
    "source": SOURCE_CITATION,
    "design": "independent_groups",
    "baseline_group": BASELINE_GROUP,
    "comparison_groups": list(COMPARISON_GROUPS),
    "measures": {
      measure: {"units": units, "description": description}
      for measure, (_column, units, description) in MEASURES.items()
    },
    "groups": groups,
    "notes": {
      "design": (
        "Independent groups: each listener appears in exactly one group, so there "
        "is no within-subject pairing and no per-subject ratio. Only ratios of "
        "group means are defined for this dataset."
      ),
      "abr_thresholds": (
        "This dataset reports no ABR thresholds, so none are included. The "
        "audiometric_*_db_hl measures are behavioural pure-tone averages in dB HL "
        "and are not a substitute for them."
      ),
      "nan_handling": (
        "Subjects carrying the literal string 'NaN' in a wave amplitude field are "
        "excluded from that measure's statistics only; each measure reports its "
        "own valid n alongside the group's total n."
      ),
      "generated_by": "scripts/generate_human_abr_summary.py",
    },
  }


def main(
  output_path: pathlib.Path = DEFAULT_OUTPUT_JSON,
  csv_path: pathlib.Path = DEFAULT_INPUT_CSV,
) -> pathlib.Path:
  """Writes the summary JSON and returns the path written."""
  summary = build_summary(csv_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  return output_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--csv", type=pathlib.Path, default=DEFAULT_INPUT_CSV)
  parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUTPUT_JSON)
  args = parser.parse_args()
  written = main(output_path=args.out, csv_path=args.csv)
  print(f"Wrote {written}.")
