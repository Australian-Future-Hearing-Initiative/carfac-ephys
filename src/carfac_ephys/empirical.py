"""Empirical chinchilla ABR data from Bharadwaj et al. (2022).

Loads the noise-exposure (temporary threshold shift) dataset shipped inside the
package (`carfac_ephys/data/`):
group ABR thresholds and high-level Wave-I / Wave-V amplitudes measured one week
before and two weeks after exposure, plus per-animal amplitudes averaged over
the 4 and 8 kHz tone bursts.

Reference: Bharadwaj et al. (2022) Commun Biol, doi:10.1038/s42003-022-03691-4.
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

# Labels of the two measurement time points in the per-animal CSV file.
PRE_TIME_POINT: str = "pre"
POST_TIME_POINT: str = "2wk"


@dataclasses.dataclass(frozen=True)
class SpeciesDataFiles:
    """File layout for one species' empirical dataset.

    Attributes:
      summary_file_name: JSON file holding group threshold and wave amplitude statistics.
      per_subject_file_name: CSV file holding per-subject pre/post wave amplitudes.
      has_wave_v: Whether the summary file reports Wave-V amplitudes in addition to Wave-I.
    """

    summary_file_name: str
    per_subject_file_name: str
    has_wave_v: bool


# Registry of supported species. Add an entry here (and the matching data
# files under `data/`) to support comparison against another species — no
# other code in this module, or in `experiment.py`, is species-specific.
SPECIES_DATA_FILES: dict[str, SpeciesDataFiles] = {
    "chinchilla": SpeciesDataFiles(
        summary_file_name="chinchilla_abr_summary.json",
        per_subject_file_name="chinABR_HighLevel_uV_4k_8k_ave.csv",
        has_wave_v=True,
    ),
    "human": SpeciesDataFiles(
        summary_file_name="human_abr_summary.json",
        per_subject_file_name="Human_Synaptopathy_ABRdata.csv",
        has_wave_v=True,
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
  """High-level ABR wave amplitudes of a single animal, in microvolts."""

  animal_id: str
  pre_w1_uv: float
  post_w1_uv: float
  pre_w5_uv: float
  post_w5_uv: float

  @property
  def w1_ratio(self) -> float:
    """Post over pre Wave-I amplitude ratio."""
    return _post_pre_ratio(self.post_w1_uv, self.pre_w1_uv, f"pre_w1_uv of {self.animal_id}")

  @property
  def w5_ratio(self) -> float:
    """Post over pre Wave-V amplitude ratio."""
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
        """Per-subject post over pre Wave-I ratio."""
        return tuple(subject.w1_ratio for subject in self.per_subject)

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


def _load_per_subject(csv_path: pathlib.Path) -> tuple[AnimalWaveAmplitudes, ...]:
    """Loads per-subject pre and post wave amplitudes from the CSV file.

    Args:
      csv_path: Path to the per-subject CSV file.

    Returns:
      Per-subject amplitudes in first-appearance order.
    """
    # Read the rows keyed by subject identifier and time point.
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")

    # Group waves by subject, preserving file order.
    waves: dict[str, dict[str, tuple[float, float]]] = {}
    for row in rows:
        subject_id = row["ID"].strip()
        time_point = row["TimePoint"].strip()
        if "W5" in row:
            waves.setdefault(subject_id, {})[time_point] = (float(row["W1"]), float(row["W5"]))
        else:
            waves.setdefault(subject_id, {})[time_point] = (float(row["W1"]), None)

    # Require both time points for every subject.
    subjects = []
    for subject_id, by_time_point in waves.items():
        missing = {PRE_TIME_POINT, POST_TIME_POINT} - set(by_time_point)
        if missing:
            raise ValueError(f"Subject {subject_id} is missing time points {sorted(missing)}.")
        pre_w1, pre_w5 = by_time_point[PRE_TIME_POINT]
        post_w1, post_w5 = by_time_point[POST_TIME_POINT]
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


def load_abr_dataset(
        species: str = DEFAULT_SPECIES,
        data_dir: pathlib.Path | str | None = None,
) -> AbrDataset:
    """Loads the empirical ABR dataset for one species.

    Args:
      species: Species key registered in `SPECIES_DATA_FILES` (e.g. "chinchilla", "human").
      data_dir: Directory holding the data files; defaults to `DEFAULT_DATA_DIR`.

    Returns:
      Parsed empirical dataset for the requested species.
    """
    if species not in SPECIES_DATA_FILES:
        raise ValueError(
            f"Unsupported species '{species}'; choose one of {sorted(SPECIES_DATA_FILES)}."
        )
    files = SPECIES_DATA_FILES[species]

    # Resolve and validate the input file paths.
    directory = pathlib.Path(DEFAULT_DATA_DIR if data_dir is None else data_dir)
    summary_path = directory / files.summary_file_name
    per_subject_path = directory / files.per_subject_file_name
    for path in (summary_path, per_subject_path):
        if not path.is_file():
            raise FileNotFoundError(f"Empirical data file not found: {path}.")

    # Parse the group summary statistics.
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frequencies = tuple(float(frequency) for frequency in summary["frequencies_hz"])
    thresholds, _ = _build_stats(summary["thresholds_db_spl"], frequencies, "thresholds_db_spl")
    w1_stats, w1_average = _build_stats(summary["high_level_w1_uv"], frequencies, "high_level_w1_uv")
    if w1_average is None:
        raise ValueError(f"{species} data: high_level_w1_uv must carry a trailing average entry.")

    # Wave-V is only reported by species registered with has_wave_v=True.
    w5_stats: dict[float, PrePostStat] | None = None
    w5_average: PrePostStat | None = None
    if files.has_wave_v:
        w5_stats, w5_average = _build_stats(
            summary["high_level_w5_uv"], frequencies, "high_level_w5_uv"
        )
        if w5_average is None:
            raise ValueError(f"{species} data: high_level_w5_uv must carry a trailing average entry.")

    # Parse the per-subject amplitudes and cross-check the subject identifiers.
    per_subject = _load_per_subject(per_subject_path)
    summary_subjects = tuple(str(s) for s in summary.get("subjects", summary.get("animals", [])))
    csv_subjects = tuple(subject.animal_id for subject in per_subject)
    if set(summary_subjects) != set(csv_subjects):
        raise ValueError(
            f"Subject identifiers disagree: summary {sorted(summary_subjects)} "
            f"vs per-subject {sorted(csv_subjects)}."
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
    )