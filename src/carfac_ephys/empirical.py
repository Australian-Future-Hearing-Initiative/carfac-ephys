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
# pyrefly: ignore [bad-argument-type]
DEFAULT_DATA_DIR: pathlib.Path = pathlib.Path(str(importlib.resources.files(__package__))) / "data"

# File names of the empirical data files within the data directory.
SUMMARY_FILE_NAME: str = "chinchilla_abr_summary.json"
PER_ANIMAL_FILE_NAME: str = "chinABR_HighLevel_uV_4k_8k_ave.csv"

# Frequency key used for the broadband click condition.
CLICK_FREQUENCY_HZ: float = 0.0

# Labels of the two measurement time points in the per-animal CSV file.
PRE_TIME_POINT: str = "pre"
POST_TIME_POINT: str = "2wk"


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


@dataclasses.dataclass(frozen=True)
class ChinchillaAbrDataset:
  """Empirical chinchilla ABR dataset before and after noise exposure.

  Attributes:
    source: Citation of the originating publication.
    animals: Animal identifiers, in file order.
    frequencies_hz: Stimulus frequencies; 0 Hz denotes the broadband click.
    thresholds_db_spl: ABR threshold statistics keyed by frequency.
    high_level_w1_uv: High-level Wave-I amplitude statistics keyed by frequency.
    high_level_w5_uv: High-level Wave-V amplitude statistics keyed by frequency.
    tone_average_w1_uv: Wave-I statistics averaged over the 4 and 8 kHz tones.
    tone_average_w5_uv: Wave-V statistics averaged over the 4 and 8 kHz tones.
    per_animal: Per-animal 4/8 kHz-averaged high-level wave amplitudes.
  """

  source: str
  animals: tuple[str, ...]
  frequencies_hz: tuple[float, ...]
  thresholds_db_spl: dict[float, PrePostStat]
  high_level_w1_uv: dict[float, PrePostStat]
  high_level_w5_uv: dict[float, PrePostStat]
  tone_average_w1_uv: PrePostStat
  tone_average_w5_uv: PrePostStat
  per_animal: tuple[AnimalWaveAmplitudes, ...]

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
    """Post over pre Wave-I ratio of the 4/8 kHz-averaged group means."""
    return self.tone_average_w1_uv.ratio

  @property
  def per_animal_w1_ratios(self) -> tuple[float, ...]:
    """Post over pre Wave-I ratio of each animal."""
    return tuple(animal.w1_ratio for animal in self.per_animal)


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


def _load_per_animal(csv_path: pathlib.Path) -> tuple[AnimalWaveAmplitudes, ...]:
  """Loads per-animal pre and post wave amplitudes from the CSV file.

  Args:
    csv_path: Path to the per-animal CSV file.

  Returns:
    Per-animal amplitudes in first-appearance order.
  """
  # Read the rows keyed by animal identifier and time point.
  with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
  if not rows:
    raise ValueError(f"No rows found in {csv_path}.")

  # Group waves by animal, preserving file order.
  waves: dict[str, dict[str, tuple[float, float]]] = {}
  for row in rows:
    animal_id = row["ID"].strip()
    time_point = row["TimePoint"].strip()
    waves.setdefault(animal_id, {})[time_point] = (float(row["W1"]), float(row["W5"]))

  # Require both time points for every animal.
  animals = []
  for animal_id, by_time_point in waves.items():
    missing = {PRE_TIME_POINT, POST_TIME_POINT} - set(by_time_point)
    if missing:
      raise ValueError(f"Animal {animal_id} is missing time points {sorted(missing)}.")
    pre_w1, pre_w5 = by_time_point[PRE_TIME_POINT]
    post_w1, post_w5 = by_time_point[POST_TIME_POINT]
    animals.append(
      AnimalWaveAmplitudes(
        animal_id=animal_id,
        pre_w1_uv=pre_w1,
        post_w1_uv=post_w1,
        pre_w5_uv=pre_w5,
        post_w5_uv=post_w5,
      )
    )
  return tuple(animals)


def load_chinchilla_abr_dataset(
  data_dir: pathlib.Path | str | None = None,
) -> ChinchillaAbrDataset:
  """Loads the Bharadwaj et al. (2022) chinchilla ABR dataset.

  Args:
    data_dir: Directory holding the data files; defaults to `DEFAULT_DATA_DIR`.

  Returns:
    Parsed empirical dataset.
  """
  # Resolve and validate the input file paths.
  directory = pathlib.Path(DEFAULT_DATA_DIR if data_dir is None else data_dir)
  summary_path = directory / SUMMARY_FILE_NAME
  per_animal_path = directory / PER_ANIMAL_FILE_NAME
  for path in (summary_path, per_animal_path):
    if not path.is_file():
      raise FileNotFoundError(f"Empirical data file not found: {path}.")

  # Parse the group summary statistics.
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  frequencies = tuple(float(frequency) for frequency in summary["frequencies_hz"])
  thresholds, _ = _build_stats(summary["thresholds_db_spl"], frequencies, "thresholds_db_spl")
  w1_stats, w1_average = _build_stats(summary["high_level_w1_uv"], frequencies, "high_level_w1_uv")
  w5_stats, w5_average = _build_stats(summary["high_level_w5_uv"], frequencies, "high_level_w5_uv")
  if w1_average is None or w5_average is None:
    raise ValueError("High-level wave blocks must carry a trailing 4/8 kHz average entry.")

  # Parse the per-animal amplitudes and cross-check the animal identifiers.
  per_animal = _load_per_animal(per_animal_path)
  summary_animals = tuple(str(animal) for animal in summary["animals"])
  csv_animals = tuple(animal.animal_id for animal in per_animal)
  if set(summary_animals) != set(csv_animals):
    raise ValueError(
      f"Animal identifiers disagree: summary {sorted(summary_animals)} "
      f"vs per-animal {sorted(csv_animals)}."
    )

  return ChinchillaAbrDataset(
    source=str(summary["source"]),
    animals=summary_animals,
    frequencies_hz=frequencies,
    thresholds_db_spl=thresholds,
    high_level_w1_uv=w1_stats,
    high_level_w5_uv=w5_stats,
    tone_average_w1_uv=w1_average,
    tone_average_w5_uv=w5_average,
    per_animal=per_animal,
  )
