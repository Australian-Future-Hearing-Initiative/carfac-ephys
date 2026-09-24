"""Empirical ABR datasets shipped as package data (`carfac_ephys/data/`).

Each registered species in `SPECIES_DATA_FILES` declares which study design its
data came from, because the two shipped datasets are not the same shape:

- `PAIRED_TIMEPOINTS` — the same subject is measured under both conditions, so
  a per-subject ratio is defined and the study reports ABR thresholds and wave
  amplitudes resolved by stimulus frequency.
- `INDEPENDENT_GROUPS` — each subject appears under exactly one condition, so
  only ratios of group means are defined; per-subject ratios do not exist, and
  the study may report no ABR thresholds at all.

`AbrDataset` therefore keeps the two kinds of quantity apart rather than
assuming every dataset has both: `ratio_of_mean_w1` is always defined,
`paired_w1_ratios` is empty unless subjects were measured twice, and
`abr_thresholds_db_spl` is `None` for a dataset that never measured them.
Measures that are not ABR quantities (behavioural audiometry, for instance)
are carried in `context_measures` with their own units, so they cannot be
mistaken for a threshold in dB SPL.

Study-specific narrative (which cohort was exposed to what, and what recovered)
belongs to the individual dataset, not to this module: it travels in the data
files' own `source` and `notes` fields.
"""

import csv
import dataclasses
import importlib.resources
import json
import math
import pathlib
from collections.abc import Sequence

# Directory holding the empirical data files shipped as package data. Resolved
# through importlib.resources so it also works from an installed wheel; assumes
# a regular (non-zipped) install, which is what pip and uv produce.
# pyrefly: ignore [bad-argument-type]
DEFAULT_DATA_DIR: pathlib.Path = pathlib.Path(str(importlib.resources.files(__package__))) / "data"

# Frequency key used for the broadband click condition.
CLICK_FREQUENCY_HZ: float = 0.0

# Study designs a dataset can declare.
PAIRED_TIMEPOINTS: str = "paired_timepoints"
INDEPENDENT_GROUPS: str = "independent_groups"
# Units accepted for ABR wave amplitudes. Calibration downstream treats these
# numbers as microvolts, so anything else is rejected rather than assumed.
WAVE_AMPLITUDE_UNITS: frozenset[str] = frozenset({"uV", "µV"})


@dataclasses.dataclass(frozen=True)
class SpeciesDataFiles:
  """Data files and comparison axis for one species' dataset.

  Attributes:
    summary_file_name: JSON file holding the group statistics.
    per_subject_file_name: CSV file holding per-subject wave amplitudes, or
      None for a dataset that needs none. An independent-groups dataset is
      fully described by its group summary, so it declares None rather than
      naming an input it never reads.
    design: `PAIRED_TIMEPOINTS` or `INDEPENDENT_GROUPS`.
    condition_column: Per-subject CSV column separating the conditions.
    baseline_label: Value of `condition_column` identifying the baseline.
    comparison_labels: Values of `condition_column` that can serve as the
      comparison condition. The first is used when none is requested.
  """

  summary_file_name: str
  per_subject_file_name: str | None
  design: str
  condition_column: str
  baseline_label: str
  comparison_labels: tuple[str, ...]


# Registry of supported species. Add an entry here, plus the matching files
# under `data/`, to compare against another dataset.
SPECIES_DATA_FILES: dict[str, SpeciesDataFiles] = {
  "chinchilla": SpeciesDataFiles(
    summary_file_name="chinchilla_abr_summary.json",
    per_subject_file_name="chinABR_HighLevel_uV_4k_8k_ave.csv",
    design=PAIRED_TIMEPOINTS,
    condition_column="TimePoint",
    baseline_label="pre",
    comparison_labels=("2wk",),
  ),
  "human": SpeciesDataFiles(
    summary_file_name="human_abr_summary.json",
    per_subject_file_name=None,
    design=INDEPENDENT_GROUPS,
    condition_column="Group",
    baseline_label="ctrl",
    comparison_labels=("nexp", "ma"),
  ),
}

# Species used when none is specified at runtime.
DEFAULT_SPECIES: str = "chinchilla"


def _ratio(comparison: float, baseline: float, label: str) -> float:
  """Returns comparison over baseline, rejecting a zero baseline."""
  if baseline == 0.0:
    raise ZeroDivisionError(f"{label} is zero; ratio is undefined.")
  return comparison / baseline


@dataclasses.dataclass(frozen=True)
class GroupComparisonStat:
  """Group mean and standard deviation for a baseline and a comparison condition.

  The two conditions are whatever the dataset's design says they are: two time
  points on the same subjects, or two independent groups of subjects.
  `difference` and `ratio_of_means` are group-level quantities in both cases,
  and are not per-subject changes.
  """

  mean_baseline: float
  mean_comparison: float
  std_baseline: float
  std_comparison: float
  n_baseline: int | None = None
  n_comparison: int | None = None
  units: str = ""

  @property
  def difference(self) -> float:
    """Comparison minus baseline difference of the group means."""
    return self.mean_comparison - self.mean_baseline

  @property
  def ratio_of_means(self) -> float:
    """Comparison over baseline ratio of the group means."""
    return _ratio(self.mean_comparison, self.mean_baseline, "mean_baseline")


@dataclasses.dataclass(frozen=True)
class SubjectWaveAmplitudes:
  """One subject's high-level ABR wave amplitudes, in microvolts.

  Only produced for a `PAIRED_TIMEPOINTS` dataset, where the same subject is
  measured under both conditions and a per-subject ratio is therefore defined.
  """

  subject_id: str
  baseline_w1_uv: float
  comparison_w1_uv: float
  baseline_w5_uv: float
  comparison_w5_uv: float

  @property
  def w1_ratio(self) -> float:
    """Comparison over baseline Wave-I amplitude ratio for this subject."""
    return _ratio(
      self.comparison_w1_uv, self.baseline_w1_uv, f"baseline_w1_uv of {self.subject_id}"
    )

  @property
  def w5_ratio(self) -> float:
    """Comparison over baseline Wave-V amplitude ratio for this subject."""
    return _ratio(
      self.comparison_w5_uv, self.baseline_w5_uv, f"baseline_w5_uv of {self.subject_id}"
    )


@dataclasses.dataclass(frozen=True)
class AbrDataset:
  """Empirical ABR dataset for one species and one chosen comparison.

  Attributes:
    species: Species key from `SPECIES_DATA_FILES`.
    source: Citation of the originating publication.
    design: `PAIRED_TIMEPOINTS` or `INDEPENDENT_GROUPS`.
    baseline_label: Condition treated as the baseline.
    comparison_group: Condition compared against the baseline.
    subjects: Subject identifiers, in file order.
    wave1_uv: Headline high-level Wave-I statistics for this comparison.
    wave5_uv: Headline Wave-V statistics, when the dataset reports Wave-V.
    frequencies_hz: Stimulus frequencies, when resolved by frequency.
    abr_thresholds_db_spl: ABR threshold statistics keyed by frequency, or
      `None` for a dataset that reports no ABR thresholds.
    frequency_wave1_uv: Wave-I statistics keyed by frequency, when available.
    frequency_wave5_uv: Wave-V statistics keyed by frequency, when available.
    per_subject: Per-subject amplitudes; empty unless the design is paired.
    context_measures: Non-ABR measures (e.g. behavioural audiometry), keyed by
      name, each carrying its own units.
  """

  species: str
  source: str
  design: str
  baseline_label: str
  comparison_group: str
  subjects: tuple[str, ...]
  wave1_uv: GroupComparisonStat
  wave5_uv: GroupComparisonStat | None = None
  frequencies_hz: tuple[float, ...] = ()
  abr_thresholds_db_spl: dict[float, GroupComparisonStat] | None = None
  frequency_wave1_uv: dict[float, GroupComparisonStat] | None = None
  frequency_wave5_uv: dict[float, GroupComparisonStat] | None = None
  per_subject: tuple[SubjectWaveAmplitudes, ...] = ()
  context_measures: dict[str, GroupComparisonStat] = dataclasses.field(default_factory=dict)

  @property
  def is_paired(self) -> bool:
    """Whether subjects were measured under both conditions."""
    return self.design == PAIRED_TIMEPOINTS

  @property
  def has_abr_thresholds(self) -> bool:
    """Whether this dataset reports ABR thresholds in dB SPL."""
    return self.abr_thresholds_db_spl is not None

  @property
  def has_wave_v(self) -> bool:
    """Whether this dataset reports Wave-V amplitudes."""
    return self.wave5_uv is not None

  @property
  def ratio_of_mean_w1(self) -> float:
    """Comparison over baseline ratio of the group-mean Wave-I amplitudes."""
    return self.wave1_uv.ratio_of_means

  @property
  def baseline_reference_w1_uv(self) -> float:
    """Baseline high-level Wave-I amplitude used to calibrate simulated units.

    For a frequency-resolved dataset this is the click condition; otherwise it
    is the headline group mean.
    """
    if self.frequency_wave1_uv is not None and CLICK_FREQUENCY_HZ in self.frequency_wave1_uv:
      return self.frequency_wave1_uv[CLICK_FREQUENCY_HZ].mean_baseline
    return self.wave1_uv.mean_baseline

  @property
  def reference_w1_ratio(self) -> float:
    """Wave-I ratio to compare a click-evoked simulation against.

    The click condition where the dataset resolves stimulus frequency, and the
    ratio of group means otherwise. Kept distinct from `ratio_of_mean_w1`,
    which is always the tone-average/group-level quantity.
    """
    if self.frequency_wave1_uv is not None and CLICK_FREQUENCY_HZ in self.frequency_wave1_uv:
      return self.frequency_wave1_uv[CLICK_FREQUENCY_HZ].ratio_of_means
    return self.ratio_of_mean_w1

  @property
  def paired_w1_ratios(self) -> tuple[float, ...]:
    """Per-subject Wave-I ratios; empty when the design is not paired."""
    return tuple(subject.w1_ratio for subject in self.per_subject)

  def threshold_shift_db(self, frequency_hz: float = CLICK_FREQUENCY_HZ) -> float:
    """Returns the comparison minus baseline ABR threshold shift in dB.

    Raises:
      ValueError: If this dataset reports no ABR thresholds.
    """
    if self.abr_thresholds_db_spl is None:
      raise ValueError(
        f"{self.species} data reports no ABR thresholds in dB SPL; "
        "check has_abr_thresholds before asking for a threshold shift."
      )
    return _lookup_frequency(self.abr_thresholds_db_spl, frequency_hz).difference

  def wave_i_ratio(self, frequency_hz: float = CLICK_FREQUENCY_HZ) -> float:
    """Returns the Wave-I ratio of group means at one stimulus frequency.

    Raises:
      ValueError: If this dataset is not resolved by stimulus frequency.
    """
    if self.frequency_wave1_uv is None:
      raise ValueError(
        f"{self.species} data is not resolved by stimulus frequency; "
        "use ratio_of_mean_w1 for the group-level Wave-I ratio."
      )
    return _lookup_frequency(self.frequency_wave1_uv, frequency_hz).ratio_of_means

  @property
  def click_threshold_shift_db(self) -> float:
    """Comparison minus baseline click ABR threshold shift in dB."""
    return self.threshold_shift_db(CLICK_FREQUENCY_HZ)


def _lookup_frequency(
  stats_by_frequency: dict[float, GroupComparisonStat],
  frequency_hz: float,
) -> GroupComparisonStat:
  """Looks up statistics for a frequency, reporting the available keys on miss."""
  key = float(frequency_hz)
  if key not in stats_by_frequency:
    available = sorted(stats_by_frequency)
    raise KeyError(f"No data for {key} Hz; available frequencies: {available}.")
  return stats_by_frequency[key]


def _build_frequency_stats(
  block: dict[str, Sequence[float]],
  frequencies_hz: Sequence[float],
  block_name: str,
  units: str,
) -> tuple[dict[float, GroupComparisonStat], GroupComparisonStat | None]:
  """Converts a frequency-resolved summary block into per-frequency statistics.

  Blocks hold one entry per frequency and, for wave amplitudes, a trailing
  entry holding the tone average. The on-disk key names (`mean_pre` and
  friends) are the paired-design wording used by the chinchilla summary file;
  they map onto the neutral baseline/comparison fields here.
  """
  keys = ("mean_pre", "mean_post", "std_pre", "std_post")
  missing = [key for key in keys if key not in block]
  if missing:
    raise ValueError(f"Block '{block_name}' is missing series {missing}.")
  lengths = {len(block[key]) for key in keys}
  if len(lengths) != 1:
    raise ValueError(f"Block '{block_name}' has series of differing lengths {sorted(lengths)}.")

  n_entries = lengths.pop()
  n_frequencies = len(frequencies_hz)
  if n_entries not in (n_frequencies, n_frequencies + 1):
    raise ValueError(
      f"Block '{block_name}' has {n_entries} entries; expected "
      f"{n_frequencies} or {n_frequencies + 1}."
    )

  def stat_at(index: int) -> GroupComparisonStat:
    return GroupComparisonStat(
      mean_baseline=float(block["mean_pre"][index]),
      mean_comparison=float(block["mean_post"][index]),
      std_baseline=float(block["std_pre"][index]),
      std_comparison=float(block["std_post"][index]),
      units=units,
    )

  stats = {float(frequency): stat_at(index) for index, frequency in enumerate(frequencies_hz)}
  aggregate = stat_at(-1) if n_entries == n_frequencies + 1 else None
  return stats, aggregate


def _row_value(row: dict[str, str], column: str) -> str:
  """Looks up a CSV field by name, ignoring case and surrounding whitespace."""
  target = column.strip().lower()
  for key, value in row.items():
    if key is not None and key.strip().lower() == target:
      return value.strip()
  raise KeyError(f"Column '{column}' not found; available columns: {list(row)}.")


def _load_paired_subjects(
  csv_path: pathlib.Path,
  files: SpeciesDataFiles,
  comparison_label: str,
) -> tuple[SubjectWaveAmplitudes, ...]:
  """Pairs the baseline and comparison rows belonging to each subject."""
  with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
  if not rows:
    raise ValueError(f"No rows found in {csv_path}.")

  waves: dict[str, dict[str, tuple[float, float]]] = {}
  for row in rows:
    subject_id = _row_value(row, "ID")
    condition = _row_value(row, files.condition_column)
    waves.setdefault(subject_id, {})[condition] = (
      float(_row_value(row, "W1")),
      float(_row_value(row, "W5")),
    )

  subjects = []
  for subject_id, by_condition in waves.items():
    missing = {files.baseline_label, comparison_label} - set(by_condition)
    if missing:
      raise ValueError(f"Subject {subject_id} is missing conditions {sorted(missing)}.")
    baseline_w1, baseline_w5 = by_condition[files.baseline_label]
    comparison_w1, comparison_w5 = by_condition[comparison_label]
    subjects.append(
      SubjectWaveAmplitudes(
        subject_id=subject_id,
        baseline_w1_uv=baseline_w1,
        comparison_w1_uv=comparison_w1,
        baseline_w5_uv=baseline_w5,
        comparison_w5_uv=comparison_w5,
      )
    )
  return tuple(subjects)


def _resolve_comparison_label(files: SpeciesDataFiles, comparison_group: str | None) -> str:
  """Returns the requested comparison label, or the species' default."""
  if comparison_group is None:
    return files.comparison_labels[0]
  if comparison_group not in files.comparison_labels:
    raise ValueError(
      f"comparison_group '{comparison_group}' is not valid for this dataset; "
      f"choose one of {files.comparison_labels}."
    )
  return comparison_group


def _validated_entry(entry: dict, group: str, measure: str) -> tuple[float, float, int]:
  """Returns (mean, std, n) from one summary entry, rejecting unusable statistics."""
  label = f"Group '{group}', measure '{measure}'"
  try:
    mean = float(entry["mean"])
    std = float(entry["std"])
    count = entry["n"]
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError(f"{label}: malformed entry ({error}).") from error
  if not math.isfinite(mean) or not math.isfinite(std):
    raise ValueError(f"{label}: mean and SD must be finite, got mean={mean}, std={std}.")
  if std < 0.0:
    raise ValueError(f"{label}: SD must not be negative, got {std}.")
  if isinstance(count, bool) or not isinstance(count, int):
    raise ValueError(f"{label}: n must be an integer, got {count!r}.")
  if count < 1:
    raise ValueError(f"{label}: n must be positive, got {count}.")
  return mean, std, count


def _group_stat(
  summary: dict,
  measure: str,
  baseline_group: str,
  comparison_group: str,
  units: str,
) -> GroupComparisonStat:
  """Builds one statistic by pairing two groups' entries from the summary."""
  groups = summary["groups"]
  for group in (baseline_group, comparison_group):
    if group not in groups:
      raise ValueError(f"Summary has no '{group}' group; found {sorted(groups)}.")
    if measure not in groups[group]:
      raise ValueError(f"Group '{group}' has no '{measure}' measure.")
  baseline_mean, baseline_std, baseline_n = _validated_entry(
    groups[baseline_group][measure], baseline_group, measure
  )
  comparison_mean, comparison_std, comparison_n = _validated_entry(
    groups[comparison_group][measure], comparison_group, measure
  )
  return GroupComparisonStat(
    mean_baseline=baseline_mean,
    mean_comparison=comparison_mean,
    std_baseline=baseline_std,
    std_comparison=comparison_std,
    n_baseline=baseline_n,
    n_comparison=comparison_n,
    units=units,
  )


def _load_paired_dataset(
  species: str,
  files: SpeciesDataFiles,
  summary: dict,
  per_subject_path: pathlib.Path,
  comparison_label: str,
) -> AbrDataset:
  """Builds a dataset from a frequency-resolved, repeated-measures summary."""
  frequencies = tuple(float(frequency) for frequency in summary["frequencies_hz"])
  thresholds, _ = _build_frequency_stats(
    summary["thresholds_db_spl"], frequencies, "thresholds_db_spl", "dB SPL"
  )
  w1_stats, w1_average = _build_frequency_stats(
    summary["high_level_w1_uv"], frequencies, "high_level_w1_uv", "uV"
  )
  w5_stats, w5_average = _build_frequency_stats(
    summary["high_level_w5_uv"], frequencies, "high_level_w5_uv", "uV"
  )
  if w1_average is None or w5_average is None:
    raise ValueError("High-level wave blocks must carry a trailing tone-average entry.")

  per_subject = _load_paired_subjects(per_subject_path, files, comparison_label)
  summary_subjects = tuple(
    str(subject) for subject in summary.get("subjects", summary.get("animals", []))
  )
  csv_subjects = tuple(subject.subject_id for subject in per_subject)
  if set(summary_subjects) != set(csv_subjects):
    raise ValueError(
      f"Subject identifiers disagree: summary {sorted(summary_subjects)} "
      f"vs per-subject {sorted(csv_subjects)}."
    )

  return AbrDataset(
    species=species,
    source=str(summary["source"]),
    design=files.design,
    baseline_label=files.baseline_label,
    comparison_group=comparison_label,
    subjects=summary_subjects,
    wave1_uv=w1_average,
    wave5_uv=w5_average,
    frequencies_hz=frequencies,
    abr_thresholds_db_spl=thresholds,
    frequency_wave1_uv=w1_stats,
    frequency_wave5_uv=w5_stats,
    per_subject=per_subject,
  )


def _load_independent_groups_dataset(
  species: str,
  files: SpeciesDataFiles,
  summary: dict,
  comparison_label: str,
) -> AbrDataset:
  """Builds a dataset from an independent-groups summary.

  No per-subject amplitudes are loaded: each subject appears under exactly one
  condition, so a per-subject ratio does not exist and the per-subject file
  adds nothing the group statistics do not already carry.
  """
  baseline_group = str(summary.get("baseline_group", files.baseline_label))
  measures = summary.get("measures", {})

  def units_for(measure: str) -> str:
    return str(measures.get(measure, {}).get("units", ""))

  def wave_units_for(measure: str) -> str:
    units = units_for(measure)
    if units not in WAVE_AMPLITUDE_UNITS:
      raise ValueError(
        f"Measure '{measure}' declares units {units!r}; wave amplitudes must be "
        f"one of {sorted(WAVE_AMPLITUDE_UNITS)}."
      )
    return units

  wave1 = _group_stat(
    summary, "wave1_uv", baseline_group, comparison_label, wave_units_for("wave1_uv")
  )
  wave5 = None
  if "wave5_uv" in summary["groups"][baseline_group]:
    wave5 = _group_stat(
      summary, "wave5_uv", baseline_group, comparison_label, wave_units_for("wave5_uv")
    )

  # Everything that is not an ABR wave amplitude is context, carried with its
  # own units so it cannot be mistaken for a threshold in dB SPL.
  context = {
    measure: _group_stat(summary, measure, baseline_group, comparison_label, units_for(measure))
    for measure in measures
    if measure not in ("wave1_uv", "wave5_uv")
  }

  return AbrDataset(
    species=species,
    source=str(summary["source"]),
    design=files.design,
    baseline_label=baseline_group,
    comparison_group=comparison_label,
    subjects=tuple(str(subject) for subject in summary.get("subjects", ())),
    wave1_uv=wave1,
    wave5_uv=wave5,
    context_measures=context,
  )


def load_abr_dataset(
  species: str = DEFAULT_SPECIES,
  data_dir: pathlib.Path | str | None = None,
  comparison_group: str | None = None,
) -> AbrDataset:
  """Loads the empirical ABR dataset for one species.

  Args:
    species: Species key registered in `SPECIES_DATA_FILES`.
    data_dir: Directory holding the data files; defaults to `DEFAULT_DATA_DIR`.
    comparison_group: Condition to compare against the baseline. Defaults to
      the species' first registered comparison label.

  Returns:
    Parsed empirical dataset for the requested comparison.
  """
  if species not in SPECIES_DATA_FILES:
    raise ValueError(
      f"Unsupported species '{species}'; choose one of {sorted(SPECIES_DATA_FILES)}."
    )
  files = SPECIES_DATA_FILES[species]
  comparison_label = _resolve_comparison_label(files, comparison_group)

  directory = pathlib.Path(DEFAULT_DATA_DIR if data_dir is None else data_dir)
  summary_path = directory / files.summary_file_name
  if not summary_path.is_file():
    raise FileNotFoundError(f"Empirical data file not found: {summary_path}.")
  summary = json.loads(summary_path.read_text(encoding="utf-8"))

  if files.design == PAIRED_TIMEPOINTS:
    if files.per_subject_file_name is None:
      raise ValueError(f"'{species}' is a paired design but declares no per-subject file.")
    per_subject_path = directory / files.per_subject_file_name
    if not per_subject_path.is_file():
      raise FileNotFoundError(f"Empirical data file not found: {per_subject_path}.")
    return _load_paired_dataset(species, files, summary, per_subject_path, comparison_label)
  return _load_independent_groups_dataset(species, files, summary, comparison_label)
