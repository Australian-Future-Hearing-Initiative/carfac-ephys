"""Empirical ABR datasets shipped inside the package (`carfac_ephys/data/`).

Loads group ABR thresholds and high-level Wave-I / Wave-V amplitude
statistics, plus per-subject wave amplitudes, for each species registered in
`SPECIES_DATA_FILES`. Two comparison designs are supported (see
`SpeciesDataFiles.paired`):

- Paired (repeated-measures): the chinchilla noise-exposure study measures
  the same animal "pre" and "2wk" after acoustic overexposure (Bharadwaj
  et al. 2022, Commun Biol, doi:10.1038/s42003-022-03691-4).
- Unpaired (independent-groups): the human synaptopathy dataset instead
  measures independent groups of listeners ("ctrl", "nexp", "ma").

Adding a new species only means registering a `SpeciesDataFiles` entry below
and the matching data files; nothing else in this module or in
`experiment.py` branches on which one is loaded.
"""

import csv
import dataclasses
import importlib.resources
import json
import pathlib
from collections.abc import Sequence

# Directory holding the empirical data files shipped as package data. Resolved
# through importlib.resources so it also works from an installed wheel; assumes
# a regular (non-zipped) install, which is what pip and uv produce.
DEFAULT_DATA_DIR: pathlib.Path = pathlib.Path(str(importlib.resources.files(__package__))) / "data"

# Frequency key used for the broadband click condition.
CLICK_FREQUENCY_HZ: float = 0.0

# Labels of chinchilla's two measurement time points in the per-animal CSV
# file; used below as that species' `baseline_label`/`comparison_labels`.
PRE_TIME_POINT: str = "pre"
POST_TIME_POINT: str = "2wk"


@dataclasses.dataclass(frozen=True)
class SpeciesDataFiles:
    """File layout and per-subject comparison axis for one species' dataset.

    Two comparison designs are supported. A *paired* (repeated-measures)
    design, like the chinchilla noise-exposure study, measures the same
    subject under both conditions (`TimePoint` "pre" and "2wk"), matched by
    ID. An *unpaired* (independent-groups) design, like the human
    synaptopathy dataset, instead measures independent groups of different
    subjects (`Group` "ctrl", "nexp", "ma"): each subject appears under
    exactly one label.

    Attributes:
      summary_file_name: JSON file holding group threshold and wave amplitude statistics.
      per_subject_file_name: CSV file holding per-subject wave amplitudes.
      has_wave_v: Whether the summary file reports Wave-V amplitudes in addition to Wave-I.
      condition_column: Per-subject CSV column that distinguishes the two
        conditions being compared (e.g. "TimePoint" or "Group").
      baseline_label: Value of `condition_column` identifying the healthy/
        baseline condition (e.g. "pre", "ctrl").
      comparison_labels: Values of `condition_column` that can stand in as the
        impaired/exposed condition (e.g. ("2wk",) or ("nexp", "ma")). The
        first entry is used by default when `load_abr_dataset` isn't told
        which one to use.
      paired: Whether `condition_column` values are repeated measures on the
        same subject (True: every subject appears once per label, matched by
        ID) or independent groups of different subjects (False: each subject
        appears under exactly one label).
    """

    summary_file_name: str
    per_subject_file_name: str
    has_wave_v: bool
    condition_column: str = "TimePoint"
    baseline_label: str = PRE_TIME_POINT
    comparison_labels: tuple[str, ...] = (POST_TIME_POINT,)
    paired: bool = True


# Registry of supported species. Add an entry here (and the matching data
# files under `data/`) to support comparison against another species — no
# other code in this module, or in `experiment.py`, is species-specific.
SPECIES_DATA_FILES: dict[str, SpeciesDataFiles] = {
    "chinchilla": SpeciesDataFiles(
        summary_file_name="chinchilla_abr_summary.json",
        per_subject_file_name="chinABR_HighLevel_uV_4k_8k_ave.csv",
        has_wave_v=True,
        condition_column="TimePoint",
        baseline_label=PRE_TIME_POINT,
        comparison_labels=(POST_TIME_POINT,),
        paired=True,
    ),
    "human": SpeciesDataFiles(
        summary_file_name="human_abr_summary.json",
        per_subject_file_name="Human_Synaptopathy_ABRdata.csv",
        has_wave_v=True,
        condition_column="Group",
        baseline_label="ctrl",
        comparison_labels=("nexp", "ma"),
        paired=False,
    ),
}

# Species used when none is specified at runtime.
DEFAULT_SPECIES: str = "chinchilla"


def _post_pre_ratio(post: float, pre: float, label: str) -> float:
  """Returns post over pre, rejecting a zero baseline.

  Args:
    post: Value measured after exposure.
    pre: Baseline value measured before exposure.
    label: Name of the baseline, used in the error message.

  Returns:
    Dimensionless post over pre ratio.
  """
  if pre == 0.0:
    raise ZeroDivisionError(f"{label} is zero; ratio is undefined.")
  return post / pre


@dataclasses.dataclass(frozen=True)
class PrePostStat:
  """Group mean and standard deviation before and after noise exposure."""

  mean_pre: float
  mean_post: float
  std_pre: float
  std_post: float

  @property
  def shift(self) -> float:
    """Post minus pre difference of the group means."""
    return self.mean_post - self.mean_pre

  @property
  def ratio(self) -> float:
    """Post over pre ratio of the group means."""
    return _post_pre_ratio(self.mean_post, self.mean_pre, "mean_pre")


@dataclasses.dataclass(frozen=True)
class AnimalWaveAmplitudes:
  """High-level ABR wave amplitudes of a single subject, in microvolts.

  For a *paired* species (e.g. chinchilla) both the baseline and comparison
  fields come from the same subject's two rows, so `w1_ratio`/`w5_ratio` are
  always defined. For an *unpaired* species (e.g. human) a subject appears
  in exactly one group, so only the pair of fields matching that subject's
  group is populated; the other pair is `None`, and the ratio properties
  raise rather than silently combining measurements from two different
  individuals — use `AbrDataset.tone_average_w1_uv.ratio` (or `wave_i_ratio`)
  for the group-level comparison instead.
  """

  animal_id: str
  pre_w1_uv: float | None
  post_w1_uv: float | None
  pre_w5_uv: float | None
  post_w5_uv: float | None

  @property
  def w1_ratio(self) -> float:
    """Post over pre Wave-I amplitude ratio.

    Raises:
      ValueError: If this subject only has one of the two measurements
        (unpaired design), so no per-subject ratio can be formed.
    """
    if self.pre_w1_uv is None or self.post_w1_uv is None:
      raise ValueError(
        f"{self.animal_id} has only one condition measured; no per-subject "
        "Wave-I ratio exists for an unpaired design."
      )
    return _post_pre_ratio(self.post_w1_uv, self.pre_w1_uv, f"pre_w1_uv of {self.animal_id}")

  @property
  def w5_ratio(self) -> float:
    """Post over pre Wave-V amplitude ratio.

    Raises:
      ValueError: If this subject only has one of the two measurements
        (unpaired design), so no per-subject ratio can be formed.
    """
    if self.pre_w5_uv is None or self.post_w5_uv is None:
      raise ValueError(
        f"{self.animal_id} has only one condition measured; no per-subject "
        "Wave-V ratio exists for an unpaired design."
      )
    return _post_pre_ratio(self.post_w5_uv, self.pre_w5_uv, f"pre_w5_uv of {self.animal_id}")


# ──────────────────────────────────────────────────────────────────────────────
# Generic protocol — any dataset (chinchilla, human, …) must satisfy this.
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class AbrDataset:
    """Empirical ABR dataset before and after noise exposure, for one species.

    A single shape serves every registered species: chinchilla data additionally
    reports Wave-V amplitudes, so `high_level_w5_uv` and `tone_average_w5_uv` are
    `None` for species that don't measure it (see `has_wave_v`).

    Attributes:
      species: Species key from `SPECIES_DATA_FILES` (e.g. "chinchilla", "human").
      source: Citation of the originating publication.
      subjects: Subject identifiers, in file order.
      frequencies_hz: Stimulus frequencies; 0 Hz denotes the broadband click.
      thresholds_db_spl: ABR threshold statistics keyed by frequency.
      high_level_w1_uv: High-level Wave-I amplitude statistics keyed by frequency.
      tone_average_w1_uv: Wave-I statistics averaged over the reference tones.
      per_subject: Per-subject high-level wave amplitudes.
      high_level_w5_uv: High-level Wave-V amplitude statistics keyed by frequency,
        when the species reports Wave-V; `None` otherwise.
      tone_average_w5_uv: Wave-V statistics averaged over the reference tones,
        when available; `None` otherwise.
      calibration_level_db: Suprathreshold reference level in dB SPL used for
        the tone-average statistics.
      comparison_group: The condition-column label used as the impaired/
        exposed group for this comparison (e.g. "2wk" for chinchilla; "nexp"
        or "ma" for human — see `SpeciesDataFiles.comparison_labels`).
      paired: Whether `per_subject` holds repeated measures on the same
        subject (True) or independent groups of different subjects (False).
        See `SpeciesDataFiles.paired`.
    """

    species: str
    source: str
    subjects: tuple[str, ...]
    frequencies_hz: tuple[float, ...]
    thresholds_db_spl: dict[float, PrePostStat]
    high_level_w1_uv: dict[float, PrePostStat]
    tone_average_w1_uv: PrePostStat
    per_subject: tuple[AnimalWaveAmplitudes, ...]
    high_level_w5_uv: dict[float, PrePostStat] | None = None
    tone_average_w5_uv: PrePostStat | None = None
    calibration_level_db: float = 80.0
    comparison_group: str = POST_TIME_POINT
    paired: bool = True

    def threshold_shift_db(self, frequency_hz: float = CLICK_FREQUENCY_HZ) -> float:
        """Returns the post minus pre threshold shift in dB at a frequency.

        Args:
          frequency_hz: Stimulus frequency; 0 Hz denotes the broadband click.

        Returns:
          Threshold shift in dB.
        """
        return _lookup_frequency(self.thresholds_db_spl, frequency_hz).shift

    def wave_i_ratio(self, frequency_hz: float = CLICK_FREQUENCY_HZ) -> float:
        """Returns the post over pre high-level Wave-I amplitude ratio.

        Args:
          frequency_hz: Stimulus frequency; 0 Hz denotes the broadband click.

        Returns:
          Dimensionless amplitude ratio.
        """
        return _lookup_frequency(self.high_level_w1_uv, frequency_hz).ratio

    @property
    def click_threshold_shift_db(self) -> float:
        """Post minus pre click ABR threshold shift in dB."""
        return self.threshold_shift_db(CLICK_FREQUENCY_HZ)

    @property
    def suprathreshold_w1_ratio(self) -> float:
        """Post over pre Wave-I ratio of the group-averaged reference tones."""
        return self.tone_average_w1_uv.ratio

    @property
    def per_animal_w1_ratios(self) -> tuple[float, ...]:
        """Per-subject post over pre Wave-I ratio.

        For an unpaired (independent-groups) species — see `paired` — no
        subject carries both measurements, so this is empty rather than
        raising; use `suprathreshold_w1_ratio` or `wave_i_ratio` for the
        group-level comparison instead.
        """
        ratios = []
        for subject in self.per_subject:
          try:
            ratios.append(subject.w1_ratio)
          except ValueError:
            continue
        return tuple(ratios)

    @property
    def has_wave_v(self) -> bool:
        """Whether this dataset reports Wave-V amplitudes."""
        return self.high_level_w5_uv is not None


def _lookup_frequency(
  stats_by_frequency: dict[float, PrePostStat],
  frequency_hz: float,
) -> PrePostStat:
  """Looks up statistics for a frequency, reporting the available keys on miss.

  Args:
    stats_by_frequency: Statistics keyed by stimulus frequency in Hz.
    frequency_hz: Frequency to look up.

  Returns:
    Statistics for the requested frequency.
  """
  key = float(frequency_hz)
  if key not in stats_by_frequency:
    available = sorted(stats_by_frequency)
    raise KeyError(f"No data for {key} Hz; available frequencies: {available}.")
  return stats_by_frequency[key]


def _build_stats(
  block: dict[str, Sequence[float]],
  frequencies_hz: Sequence[float],
  block_name: str,
) -> tuple[dict[float, PrePostStat], PrePostStat | None]:
  """Converts a summary statistics block into per-frequency statistics.

  Blocks hold one entry per frequency and, for wave amplitudes, a trailing entry
  holding the 4/8 kHz average.

  Args:
    block: Mapping of 'mean_pre', 'mean_post', 'std_pre', 'std_post' to values.
    frequencies_hz: Stimulus frequencies in Hz.
    block_name: Block name, used in error messages.

  Returns:
    Per-frequency statistics and the trailing aggregate, if present.
  """
  # Validate that all required series are present and equally long.
  keys = ("mean_pre", "mean_post", "std_pre", "std_post")
  missing = [key for key in keys if key not in block]
  if missing:
    raise ValueError(f"Block '{block_name}' is missing series {missing}.")
  lengths = {len(block[key]) for key in keys}
  if len(lengths) != 1:
    raise ValueError(f"Block '{block_name}' has series of differing lengths {sorted(lengths)}.")

  # Validate the entry count against the frequency count.
  n_entries = lengths.pop()
  n_frequencies = len(frequencies_hz)
  if n_entries not in (n_frequencies, n_frequencies + 1):
    raise ValueError(
      f"Block '{block_name}' has {n_entries} entries; expected "
      f"{n_frequencies} or {n_frequencies + 1}."
    )

  # Build per-frequency statistics plus the trailing aggregate, if present.
  def stat_at(index: int) -> PrePostStat:
    return PrePostStat(
      mean_pre=float(block["mean_pre"][index]),
      mean_post=float(block["mean_post"][index]),
      std_pre=float(block["std_pre"][index]),
      std_post=float(block["std_post"][index]),
    )

  stats = {float(frequency): stat_at(index) for index, frequency in enumerate(frequencies_hz)}
  aggregate = stat_at(-1) if n_entries == n_frequencies + 1 else None
  return stats, aggregate


def _row_value(row: dict[str, str], column: str) -> str:
  """Looks up a CSV field by name, ignoring case (chinchilla's "W1"/"W5" vs
  human's "w1"/"w5") and surrounding whitespace.

  Args:
    row: One `csv.DictReader` row.
    column: Column name to look up, matched case-insensitively.

  Returns:
    The stripped field value.

  Raises:
    KeyError: If no column matches, listing the columns that were found.
  """
  target = column.strip().lower()
  for key, value in row.items():
    if key is not None and key.strip().lower() == target:
      return value.strip()
  raise KeyError(f"Column '{column}' not found; available columns: {list(row)}.")


def _row_wave_amplitudes(row: dict[str, str]) -> tuple[float, float | None]:
  """Extracts (W1, W5) from a row, tolerating a missing Wave-V column."""
  w1 = float(_row_value(row, "W1"))
  try:
    w5 = float(_row_value(row, "W5"))
  except KeyError:
    w5 = None
  return w1, w5


def _load_paired_per_subject(
  rows: list[dict[str, str]],
  files: SpeciesDataFiles,
  comparison_label: str,
) -> tuple[AnimalWaveAmplitudes, ...]:
  """Pairs baseline and comparison rows belonging to the same subject ID.

  Used for a repeated-measures species (`files.paired` is True), e.g.
  chinchilla's "pre" vs "2wk" TimePoint rows for the same animal.
  """
  # Group waves by subject, preserving file order.
  waves: dict[str, dict[str, tuple[float, float | None]]] = {}
  for row in rows:
    subject_id = _row_value(row, "ID")
    condition = _row_value(row, files.condition_column)
    waves.setdefault(subject_id, {})[condition] = _row_wave_amplitudes(row)

  # Require both conditions for every subject.
  subjects = []
  for subject_id, by_condition in waves.items():
    missing = {files.baseline_label, comparison_label} - set(by_condition)
    if missing:
      raise ValueError(f"Subject {subject_id} is missing time points {sorted(missing)}.")
    pre_w1, pre_w5 = by_condition[files.baseline_label]
    post_w1, post_w5 = by_condition[comparison_label]
    subjects.append(
      AnimalWaveAmplitudes(
        animal_id=subject_id,
        pre_w1_uv=pre_w1,
        post_w1_uv=post_w1,
        pre_w5_uv=pre_w5,
        post_w5_uv=post_w5,
      )
    )
  return tuple(subjects)


def _load_grouped_per_subject(
  rows: list[dict[str, str]],
  files: SpeciesDataFiles,
  comparison_label: str,
) -> tuple[AnimalWaveAmplitudes, ...]:
  """Keeps subjects belonging to either of two independent groups.

  Used for an independent-groups species (`files.paired` is False), e.g.
  human "ctrl" vs "nexp" (or "ma") Group rows, where each subject appears
  under exactly one label. Rows in neither group (e.g. "ma" rows when
  comparing against "nexp") are skipped. Only the fields matching a
  subject's own group are populated; see `AnimalWaveAmplitudes`.
  """
  subjects = []
  seen_ids: set[str] = set()
  for row in rows:
    condition = _row_value(row, files.condition_column)
    if condition not in (files.baseline_label, comparison_label):
      continue
    subject_id = _row_value(row, "ID")
    if subject_id in seen_ids:
      raise ValueError(
        f"Subject {subject_id} appears more than once for an unpaired species."
      )
    seen_ids.add(subject_id)
    w1, w5 = _row_wave_amplitudes(row)
    is_baseline = condition == files.baseline_label
    subjects.append(
      AnimalWaveAmplitudes(
        animal_id=subject_id,
        pre_w1_uv=w1 if is_baseline else None,
        post_w1_uv=None if is_baseline else w1,
        pre_w5_uv=w5 if is_baseline else None,
        post_w5_uv=None if is_baseline else w5,
      )
    )
  return tuple(subjects)


def _select_group_block(
  summary_block: dict,
  files: SpeciesDataFiles,
  comparison_label: str,
  block_name: str,
) -> dict[str, Sequence[float]]:
  """Selects the right group-statistics block from a summary section.

  A species with only one `comparison_labels` option (e.g. chinchilla) keeps
  its summary blocks flat: `{"mean_pre": [...], "mean_post": [...], ...}`,
  since there's only one possible comparison. A species with more than one
  option (e.g. human's "nexp"/"ma") must instead nest a block per comparison
  label: `{"nexp": {"mean_pre": [...], ...}, "ma": {"mean_pre": [...], ...}}`
  — the baseline ("ctrl") side is shared, but the comparison side (and thus
  `mean_post`/`std_post`) differs depending on which group was requested, so
  a single flat block can't represent both.

  Args:
    summary_block: The raw JSON value for this section (`thresholds_db_spl`,
      `high_level_w1_uv`, or `high_level_w5_uv`).
    files: File layout and comparison-axis configuration for this species.
    comparison_label: The resolved comparison label for this load.
    block_name: Block name, used in error messages.

  Returns:
    The flat `{"mean_pre": [...], "mean_post": [...], ...}` mapping to build
    statistics from.
  """
  if len(files.comparison_labels) <= 1:
    return summary_block
  if comparison_label not in summary_block:
    raise ValueError(
      f"Block '{block_name}' has no '{comparison_label}' entry; expected one "
      f"nested block per comparison label: {files.comparison_labels}."
    )
  return summary_block[comparison_label]


def _load_per_subject(
  csv_path: pathlib.Path,
  files: SpeciesDataFiles,
  comparison_label: str,
) -> tuple[AnimalWaveAmplitudes, ...]:
    """Loads per-subject wave amplitudes from the CSV file.

    Dispatches on `files.paired`: a repeated-measures species requires every
    subject to carry both `files.baseline_label` and `comparison_label` rows,
    matched by ID (`_load_paired_per_subject`); an independent-groups species
    instead keeps every subject who falls in either group, with only the
    matching pair of fields populated (`_load_grouped_per_subject`).

    Args:
      csv_path: Path to the per-subject CSV file.
      files: File layout and comparison-axis configuration for this species.
      comparison_label: The `files.condition_column` value standing in for
        the impaired/exposed condition (one of `files.comparison_labels`).

    Returns:
      Per-subject amplitudes in first-appearance order.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")

    if files.paired:
      return _load_paired_per_subject(rows, files, comparison_label)
    return _load_grouped_per_subject(rows, files, comparison_label)


def load_abr_dataset(
        species: str = DEFAULT_SPECIES,
        data_dir: pathlib.Path | str | None = None,
        comparison_group: str | None = None,
) -> AbrDataset:
    """Loads the empirical ABR dataset for one species.

    Args:
      species: Species key registered in `SPECIES_DATA_FILES` (e.g. "chinchilla", "human").
      data_dir: Directory holding the data files; defaults to `DEFAULT_DATA_DIR`.
      comparison_group: Which of the species' `comparison_labels` to treat as
        the impaired/exposed condition (e.g. "nexp" or "ma" for human).
        Defaults to the first registered label (chinchilla: "2wk"; human:
        "nexp"). A species with only one comparison label (chinchilla) has
        nothing to choose between, so this is rarely needed outside "human".

    Returns:
      Parsed empirical dataset for the requested species.
    """
    if species not in SPECIES_DATA_FILES:
        raise ValueError(
            f"Unsupported species '{species}'; choose one of {sorted(SPECIES_DATA_FILES)}."
        )
    files = SPECIES_DATA_FILES[species]

    # Resolve which comparison label to compare against the baseline.
    if comparison_group is None:
        comparison_label = files.comparison_labels[0]
    elif comparison_group in files.comparison_labels:
        comparison_label = comparison_group
    else:
        raise ValueError(
            f"comparison_group '{comparison_group}' is not valid for '{species}'; "
            f"choose one of {files.comparison_labels}."
        )

    # Resolve and validate the input file paths.
    directory = pathlib.Path(DEFAULT_DATA_DIR if data_dir is None else data_dir)
    summary_path = directory / files.summary_file_name
    per_subject_path = directory / files.per_subject_file_name
    for path in (summary_path, per_subject_path):
        if not path.is_file():
            raise FileNotFoundError(f"Empirical data file not found: {path}.")

    # Parse the group summary statistics. `_select_group_block` picks the
    # right (possibly comparison_label-nested) block so a species with more
    # than one comparison_labels option — e.g. human's "nexp"/"ma" — reports
    # the statistics for whichever one was actually requested, not always
    # the same hard-coded pair.
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frequencies = tuple(float(frequency) for frequency in summary["frequencies_hz"])
    thresholds, _ = _build_stats(
        _select_group_block(summary["thresholds_db_spl"], files, comparison_label, "thresholds_db_spl"),
        frequencies,
        "thresholds_db_spl",
    )
    w1_stats, w1_average = _build_stats(
        _select_group_block(summary["high_level_w1_uv"], files, comparison_label, "high_level_w1_uv"),
        frequencies,
        "high_level_w1_uv",
    )
    if w1_average is None:
        raise ValueError(f"{species} data: high_level_w1_uv must carry a trailing average entry.")

    # Wave-V is only reported by species registered with has_wave_v=True.
    w5_stats: dict[float, PrePostStat] | None = None
    w5_average: PrePostStat | None = None
    if files.has_wave_v:
        w5_stats, w5_average = _build_stats(
            _select_group_block(summary["high_level_w5_uv"], files, comparison_label, "high_level_w5_uv"),
            frequencies,
            "high_level_w5_uv",
        )
        if w5_average is None:
            raise ValueError(f"{species} data: high_level_w5_uv must carry a trailing average entry.")

    # Parse the per-subject amplitudes and cross-check the subject identifiers.
    per_subject = _load_per_subject(per_subject_path, files, comparison_label)
    summary_subjects = tuple(str(s) for s in summary.get("subjects", summary.get("animals", [])))
    csv_subjects = tuple(subject.animal_id for subject in per_subject)
    if files.paired:
        # Repeated measures on a small cohort: every subject must appear on
        # both sides, so the two ID sets must match exactly.
        if set(summary_subjects) != set(csv_subjects):
            raise ValueError(
                f"Subject identifiers disagree: summary {sorted(summary_subjects)} "
                f"vs per-subject {sorted(csv_subjects)}."
            )
    else:
        # Independent groups: the summary may list subjects from groups not
        # selected as this comparison_group (e.g. "ma" while comparing
        # against "nexp"), so only require that every subject kept from the
        # CSV is accounted for in the summary.
        unknown = set(csv_subjects) - set(summary_subjects)
        if unknown:
            raise ValueError(
                f"Per-subject identifiers not listed in summary: {sorted(unknown)}."
            )

    return AbrDataset(
        species=species,
        source=str(summary["source"]),
        subjects=summary_subjects,
        frequencies_hz=frequencies,
        thresholds_db_spl=thresholds,
        high_level_w1_uv=w1_stats,
        tone_average_w1_uv=w1_average,
        per_subject=per_subject,
        high_level_w5_uv=w5_stats,
        tone_average_w5_uv=w5_average,
        calibration_level_db=float(summary.get("calibration_level_db", 80.0)),
        comparison_group=comparison_label,
        paired=files.paired,
    )
