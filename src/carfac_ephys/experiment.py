"""Cohort definitions and simulation level series runners for CARFAC electrophysiology."""

from collections.abc import Mapping, Sequence
import pathlib
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from carfac_ephys import carfac_model
from carfac_ephys import constants
from carfac_ephys import electrophysiology
from carfac_ephys import stimuli


class CohortCondition(NamedTuple):
  """Parameters defining a cohort biophysical condition."""

  name: str
  ohc_health: float = 1.0
  fiber_retention: float | carfac_model.FiberRetention = 1.0
  description: str = ""


DEFAULT_COHORT_CONDITIONS: tuple[CohortCondition, ...] = (
  CohortCondition(
    name="Control",
    ohc_health=1.0,
    fiber_retention=1.0,
    description="Healthy control (100% OHC, 100% AN fibers)",
  ),
  CohortCondition(
    name="Synaptopathy-50",
    ohc_health=1.0,
    fiber_retention=0.5,
    description="50% auditory nerve deafferentation (100% OHC, 50% fibers)",
  ),
  CohortCondition(
    name="Synaptopathy-25",
    ohc_health=1.0,
    fiber_retention=0.25,
    description="75% auditory nerve deafferentation (100% OHC, 25% fibers)",
  ),
  CohortCondition(
    name="OHC-Loss",
    ohc_health=0.4,
    fiber_retention=1.0,
    description="Outer hair cell loss (40% OHC, 100% fibers)",
  ),
  CohortCondition(
    name="Mixed-Loss",
    ohc_health=0.4,
    fiber_retention=0.5,
    description="Mixed impairment (40% OHC, 50% fibers)",
  ),
)

DEFAULT_CLICK_LEVELS_DB: tuple[float, ...] = (30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
DEFAULT_EFR_LEVELS_DB: tuple[float, ...] = (40.0, 50.0, 60.0, 70.0, 80.0)


def _normalize_cohort_condition(
  name: str,
  item: CohortCondition | tuple[Any, ...] | Mapping[str, Any] | Any,
) -> CohortCondition:
  """Normalizes various condition specifications into a CohortCondition."""
  # Handle existing CohortCondition instance.
  if isinstance(item, CohortCondition):
    return item if not name or name == item.name else item._replace(name=name)

  # Handle tuple of (ohc_health, fiber_retention) or (ohc, fib, desc).
  if isinstance(item, (tuple, list)):
    if len(item) == 2:
      return CohortCondition(name=name, ohc_health=float(item[0]), fiber_retention=float(item[1]))
    if len(item) == 3:
      return CohortCondition(
        name=name,
        ohc_health=float(item[0]),
        fiber_retention=float(item[1]),
        description=str(item[2]),
      )
    raise ValueError(f"Condition tuple must have length 2 or 3, got {len(item)} items.")

  # Handle dictionary representation.
  if isinstance(item, Mapping):
    cond_name = str(item.get("name", name))
    ohc = float(item.get("ohc_health", 1.0))
    fib = item.get("fiber_retention", 1.0)
    desc = str(item.get("description", ""))
    return CohortCondition(name=cond_name, ohc_health=ohc, fiber_retention=fib, description=desc)

  # Handle numeric scalar as fiber retention factor.
  if isinstance(item, (int, float, np.number)) and not isinstance(item, (bool, np.bool_)):
    return CohortCondition(name=name, ohc_health=1.0, fiber_retention=float(item))

  raise TypeError(f"Cannot parse cohort condition '{name}' from {type(item).__name__}.")


class Cohort(dict[str, CohortCondition]):
  """Collection of experimental cohort conditions."""

  def __init__(
    self,
    conditions: (
      Sequence[CohortCondition | Mapping[str, Any] | tuple[float, float]]
      | Mapping[str, CohortCondition | Mapping[str, Any] | tuple[float, float] | float]
      | None
    ) = None,
  ) -> None:
    super().__init__()
    # Use default cohort conditions if None.
    if conditions is None:
      for cond in DEFAULT_COHORT_CONDITIONS:
        self[cond.name] = cond
      return

    # Populate from mapping.
    if isinstance(conditions, Mapping):
      for name, item in conditions.items():
        cond = _normalize_cohort_condition(str(name), item)
        self[cond.name] = cond
      return

    # Populate from sequence.
    if isinstance(conditions, Sequence):
      for idx, item in enumerate(conditions):
        fallback_name = getattr(item, "name", f"Condition-{idx + 1}")
        cond = _normalize_cohort_condition(str(fallback_name), item)
        self[cond.name] = cond
      return

    raise TypeError(f"Unsupported cohort specification type: {type(conditions).__name__}.")

  @property
  def conditions(self) -> list[CohortCondition]:
    """Returns list of cohort conditions."""
    return list(self.values())

  def __getitem__(self, key: str) -> CohortCondition:
    """Returns condition by name, supporting Control alias."""
    str_key = str(key)
    if super().__contains__(str_key):
      return super().__getitem__(str_key)
    # Support alias between "Control" and "Control (Healthy)".
    if str_key == "Control (Healthy)" and super().__contains__("Control"):
      return super().__getitem__("Control")
    if str_key == "Control" and super().__contains__("Control (Healthy)"):
      return super().__getitem__("Control (Healthy)")
    return super().__getitem__(str_key)

  def __contains__(self, key: object) -> bool:
    """Checks condition membership by name, supporting Control alias."""
    if super().__contains__(key):
      return True
    # Support alias membership checks.
    if key == "Control (Healthy)" and super().__contains__("Control"):
      return True
    if key == "Control" and super().__contains__("Control (Healthy)"):
      return True
    return False


def get_default_cohort() -> Cohort:
  """Returns a new default Cohort instance with the 5 standard conditions."""
  return Cohort()


def _resolve_cohort(
  cohort: Cohort | Mapping[str, Any] | Sequence[CohortCondition] | None,
) -> Cohort:
  """Resolves cohort input argument into a Cohort instance."""
  if cohort is None:
    return Cohort()
  if isinstance(cohort, Cohort):
    return cohort
  return Cohort(cohort)


def simulate_abr_level_series(
  cohort: Cohort | Mapping[str, Any] | Sequence[CohortCondition] | None = None,
  click_levels_db: Sequence[float] = DEFAULT_CLICK_LEVELS_DB,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  duration_s: float = 0.03,
  pulse_width_s: float = 0.0001,
  delay_s: float = 0.005,
  window_s: float = 0.008,
  mode: str = "baseline_to_peak",
) -> dict[str, list[float]]:
  """Runs ABR Wave-I click level series across cohort conditions.

  Args:
    cohort: Cohort instance or mapping of conditions. Defaults to standard 5-condition cohort.
    click_levels_db: Sequence of click sound levels in dB SPL.
    sample_rate: Sampling rate in Hz.
    duration_s: Click duration in seconds.
    pulse_width_s: Click pulse width in seconds.
    delay_s: Stimulus onset delay in seconds.
    window_s: Analysis window duration in seconds.
    mode: Wave-I extraction mode ('baseline_to_peak' or 'peak_to_trough').

  Returns:
    Dictionary mapping condition name to list of Wave-I onset amplitudes.
  """
  # Validate input parameters.
  if sample_rate <= 0:
    raise ValueError("sample_rate must be positive.")
  if not click_levels_db:
    raise ValueError("click_levels_db sequence cannot be empty.")

  # Resolve cohort specifications.
  resolved_cohort = _resolve_cohort(cohort)
  results: dict[str, list[float]] = {}

  # Iterate through each condition.
  for condition in resolved_cohort.values():
    model = carfac_model.build_model(
      ohc_health=condition.ohc_health,
      fiber_retention=condition.fiber_retention,
      fs=sample_rate,
    )
    amps: list[float] = []

    # Run level sweep.
    for level in click_levels_db:
      model.reset()
      waveform = stimuli.generate_click(
        duration_s=duration_s,
        sample_rate=sample_rate,
        peak_db_spl=float(level),
        pulse_width_s=pulse_width_s,
        delay_s=delay_s,
      )
      naps = model.run(waveform)
      pop_rate = electrophysiology.compute_population_rate(naps)
      amp = electrophysiology.extract_wave_i_amplitude(
        pop_rate,
        sample_rate=sample_rate,
        stimulus_onset_s=delay_s,
        window_s=window_s,
        mode=mode,
      )
      amps.append(float(amp))

    results[condition.name] = amps

  return results


def simulate_efr_level_series(
  cohort: Cohort | Mapping[str, Any] | Sequence[CohortCondition] | None = None,
  efr_levels_db: Sequence[float] = DEFAULT_EFR_LEVELS_DB,
  fc_hz: float = 2000.0,
  fm_hz: float = 100.0,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  duration_s: float = 0.2,
  steady_state_start_s: float = 0.05,
  single_sided: bool = False,
) -> dict[str, list[float]]:
  """Runs EFR SAM tone level series across cohort conditions.

  Args:
    cohort: Cohort instance or mapping of conditions. Defaults to standard 5-condition cohort.
    efr_levels_db: Sequence of SAM tone carrier sound levels in dB SPL.
    fc_hz: Carrier frequency in Hz.
    fm_hz: Modulation frequency in Hz.
    sample_rate: Sampling rate in Hz.
    duration_s: SAM tone duration in seconds.
    steady_state_start_s: Steady-state window start time in seconds.
    single_sided: Whether to return single-sided Fourier magnitude.

  Returns:
    Dictionary mapping condition name to list of EFR spectral magnitudes.
  """
  # Validate input parameters.
  if sample_rate <= 0:
    raise ValueError("sample_rate must be positive.")
  if fc_hz <= 0.0 or fm_hz <= 0.0:
    raise ValueError("fc_hz and fm_hz must be positive.")
  if fm_hz >= sample_rate / 2.0:
    raise ValueError("fm_hz must be below Nyquist frequency.")
  if not efr_levels_db:
    raise ValueError("efr_levels_db sequence cannot be empty.")

  # Resolve cohort specifications.
  resolved_cohort = _resolve_cohort(cohort)
  results: dict[str, list[float]] = {}

  # Iterate through each condition.
  for condition in resolved_cohort.values():
    model = carfac_model.build_model(
      ohc_health=condition.ohc_health,
      fiber_retention=condition.fiber_retention,
      fs=sample_rate,
    )
    mags: list[float] = []

    # Run level sweep.
    for level in efr_levels_db:
      model.reset()
      waveform = stimuli.generate_sam_tone(
        fc_hz=fc_hz,
        fm_hz=fm_hz,
        duration_s=duration_s,
        sample_rate=sample_rate,
        target_db_spl=float(level),
      )
      naps = model.run(waveform)
      pop_rate = electrophysiology.compute_population_rate(naps)
      mag = electrophysiology.extract_efr_amplitude(
        pop_rate,
        sample_rate=sample_rate,
        fm_hz=fm_hz,
        steady_state_start_s=steady_state_start_s,
        single_sided=single_sided,
      )
      mags.append(float(mag))

    results[condition.name] = mags

  return results


def format_ascii_table(
  title: str,
  levels_db: Sequence[float],
  results: Mapping[str, Sequence[float]],
  unit: str = "dB SPL",
) -> str:
  """Formats simulation growth results into a clean ASCII table.

  Args:
    title: Table title header.
    levels_db: Sound levels in dB SPL.
    results: Mapping of condition name to sequence of amplitude values.
    unit: Unit label for sound levels.

  Returns:
    Formatted multi-line ASCII table string.
  """
  # Compute column widths.
  cond_names = list(results.keys())
  cond_width = max([len("Condition"), *(len(name) for name in cond_names)]) + 2
  level_headers = [f"{lvl:g} {unit}".strip() for lvl in levels_db]
  col_width = max([11, *(len(h) + 2 for h in level_headers)])

  # Format horizontal borders.
  table_width = cond_width + 1 + len(levels_db) * (col_width + 1)
  total_width = max(table_width, len(title))
  border = "=" * total_width
  header_sep = "-" * cond_width + "+" + "+".join(["-" * col_width] * len(levels_db))

  # Build header row.
  header_row = f"{'Condition':<{cond_width}}|" + "|".join(
    f"{h:^{col_width}}" for h in level_headers
  )

  # Build data rows.
  data_rows: list[str] = []
  for name in cond_names:
    vals = results[name]
    val_strs = [f"{v:>{col_width - 2}.4f}  " for v in vals]
    data_rows.append(f"{name:<{cond_width}}|" + "|".join(val_strs))

  lines = [
    border,
    title,
    border,
    header_row,
    header_sep,
    *data_rows,
    border,
  ]
  return "\n".join(lines)


# Visual styling specifications for cohort conditions.
COHORT_STYLES: dict[str, dict[str, Any]] = {
  "Control": {
    "color": "#111111",
    "marker": "o",
    "linestyle": "-",
    "label": "Control (100% OHC, 100% Fibers)",
  },
  "Synaptopathy-50": {
    "color": "#1f77b4",
    "marker": "s",
    "linestyle": "--",
    "label": "Synaptopathy 50% (50% Fibers)",
  },
  "Synaptopathy-25": {
    "color": "#17becf",
    "marker": "^",
    "linestyle": ":",
    "label": "Synaptopathy 25% (25% Fibers)",
  },
  "OHC-Loss": {
    "color": "#d62728",
    "marker": "d",
    "linestyle": "-.",
    "label": "OHC Loss (40% OHC, 100% Fibers)",
  },
  "Mixed-Loss": {
    "color": "#ff7f0e",
    "marker": "x",
    "linestyle": ":",
    "label": "Mixed Loss (40% OHC, 50% Fibers)",
  },
}


def _plot_growth_curves(
  levels_db: Sequence[float],
  results: Mapping[str, Sequence[float]],
  output_path: str | pathlib.Path,
  xlabel: str,
  ylabel: str,
  title: str,
) -> pathlib.Path:
  """Plots and saves cohort response growth curves across sound levels."""
  path = pathlib.Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)

  # Create figure.
  fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

  # Plot each condition curve.
  for name, values in results.items():
    style = COHORT_STYLES.get(
      name,
      {"color": "#555555", "marker": ".", "linestyle": "-", "label": name},
    )
    ax.plot(
      levels_db,
      values,
      label=style["label"],
      color=style["color"],
      marker=style["marker"],
      linestyle=style["linestyle"],
      linewidth=2.0,
      markersize=6.5,
    )

  # Labeling and formatting.
  ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
  ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
  ax.set_title(title, fontsize=12, fontweight="bold")
  ax.grid(True, linestyle="--", alpha=0.5)
  ax.legend(frameon=True, fontsize=9, loc="upper left")

  # Render and save figure.
  fig.tight_layout()
  fig.savefig(path, dpi=300)
  plt.close(fig)

  return path


def plot_abr_growth(
  click_levels_db: Sequence[float],
  results: Mapping[str, Sequence[float]],
  output_path: str | pathlib.Path,
) -> pathlib.Path:
  """Plots and saves ABR Wave-I input-output functions across cohort conditions."""
  # Render ABR Wave-I growth curves.
  return _plot_growth_curves(
    levels_db=click_levels_db,
    results=results,
    output_path=output_path,
    xlabel="Click Sound Level (dB SPL)",
    ylabel="ABR Wave-I Onset Amplitude (spikes/s)",
    title="ABR Wave-I Input-Output Growth Functions",
  )


def plot_efr_growth(
  efr_levels_db: Sequence[float],
  results: Mapping[str, Sequence[float]],
  output_path: str | pathlib.Path,
) -> pathlib.Path:
  """Plots and saves EFR spectral magnitude growth functions across conditions."""
  # Render EFR growth curves.
  return _plot_growth_curves(
    levels_db=efr_levels_db,
    results=results,
    output_path=output_path,
    xlabel="SAM Tone Sound Level (dB SPL)",
    ylabel="EFR Spectral Magnitude at 100 Hz (spikes/s)",
    title="Envelope Following Response (EFR) Growth Functions",
  )
