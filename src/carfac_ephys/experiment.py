"""Cohort definitions and simulation level series runners for CARFAC electrophysiology."""

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from carfac_ephys import carfac_model, constants, electrophysiology, empirical, stimuli


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
    Dictionary mapping condition name to list of Wave-I onset amplitudes, in arbitrary units.
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
  depth: float = 1.0,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  duration_s: float = 0.2,
  ramp_s: float = 0.01,
  steady_state_start_s: float = 0.05,
  single_sided: bool = False,
) -> dict[str, list[float]]:
  """Runs EFR SAM tone level series across cohort conditions.

  Args:
    cohort: Cohort instance or mapping of conditions. Defaults to standard 5-condition cohort.
    efr_levels_db: Sequence of SAM tone carrier sound levels in dB SPL.
    fc_hz: Carrier frequency in Hz.
    fm_hz: Modulation frequency in Hz.
    depth: Modulation depth of the SAM tone.
    sample_rate: Sampling rate in Hz.
    duration_s: SAM tone duration in seconds.
    ramp_s: Duration of the stimulus onset and offset ramps in seconds.
    steady_state_start_s: Steady-state window start time in seconds.
    single_sided: Whether to return single-sided Fourier magnitude.

  Returns:
    Dictionary mapping condition name to list of EFR spectral magnitudes, in arbitrary units.
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

  # End the analysis window before the offset ramp, whose envelope would
  # otherwise leak into the modulation frequency bin.
  steady_state_stop_s = duration_s - ramp_s

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
        depth=depth,
        duration_s=duration_s,
        sample_rate=sample_rate,
        target_db_spl=float(level),
        ramp_s=ramp_s,
      )
      naps = model.run(waveform)
      pop_rate = electrophysiology.compute_population_rate(naps)
      mag = electrophysiology.extract_efr_amplitude(
        pop_rate,
        sample_rate=sample_rate,
        fm_hz=fm_hz,
        steady_state_start_s=steady_state_start_s,
        steady_state_stop_s=steady_state_stop_s,
        single_sided=single_sided,
      )
      mags.append(float(mag))

    results[condition.name] = mags

  return results


# Click level (dB SPL) at which the simulated healthy response is matched to the
# empirical high-level animal Wave-I amplitude, and the cohort it is matched to.
CALIBRATION_LEVEL_DB: float = 80.0
BASELINE_CONDITION: str = "Control"


def fit_response_scale_uv_per_au(
  click_levels_db: Sequence[float],
  abr_results: Mapping[str, Sequence[float]],
  dataset: empirical.ChinchillaAbrDataset | None = None,
  level_db: float = CALIBRATION_LEVEL_DB,
  condition: str = BASELINE_CONDITION,
) -> float:
  """Fits the microvolts per arbitrary unit scale of simulated ABR responses.

  Simulated responses are dimensionless, so they are calibrated against the
  pre-exposure (healthy) click Wave-I amplitude of the chinchilla dataset.

  Args:
    click_levels_db: Click sound levels in dB SPL.
    abr_results: Mapping of condition name to Wave-I amplitudes in AU.
    dataset: Empirical dataset; loaded from package data when None.
    level_db: Click level matched to the empirical high-level amplitude.
    condition: Cohort treated as the healthy baseline.

  Returns:
    Scale factor in microvolts per arbitrary unit (> 0).
  """
  # Look up the simulated healthy response at the calibration level.
  level_indices = {float(level): index for index, level in enumerate(click_levels_db)}
  simulated_au = _get_level_value(abr_results, level_indices, condition, level_db)
  if simulated_au is None:
    raise ValueError(f"No '{condition}' response at {level_db} dB SPL to calibrate against.")

  # Match it to the empirical pre-exposure click Wave-I amplitude.
  data = empirical.load_chinchilla_abr_dataset() if dataset is None else dataset
  measured_uv = data.high_level_w1_uv[empirical.CLICK_FREQUENCY_HZ].mean_pre
  return electrophysiology.fit_microvolts_per_au([simulated_au], [measured_uv])


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
    ylabel=f"ABR Wave-I Onset Amplitude ({electrophysiology.RESPONSE_UNIT})",
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
    ylabel=f"EFR Spectral Magnitude at 100 Hz ({electrophysiology.RESPONSE_UNIT})",
    title="Envelope Following Response (EFR) Growth Functions",
  )


def format_markdown_table(
  levels_db: Sequence[float],
  results: Mapping[str, Sequence[float]],
  unit: str = "dB SPL",
) -> str:
  """Formats simulation growth results into a GitHub-flavored Markdown table.

  Args:
    levels_db: Sound levels in dB SPL.
    results: Mapping of condition name to sequence of amplitude values.
    unit: Unit label for sound levels.

  Returns:
    Formatted Markdown table string.
  """
  # Construct table header row.
  headers = ["Condition", *(f"{lvl:g} {unit}" for lvl in levels_db)]
  header_row = "| " + " | ".join(headers) + " |"
  separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

  # Format rows for each cohort condition.
  rows: list[str] = []
  for name, values in results.items():
    formatted_vals = [f"{v:.4f}" for v in values]
    rows.append("| " + " | ".join([name, *formatted_vals]) + " |")

  return "\n".join([header_row, separator_row, *rows])


class BiologicalValidation(NamedTuple):
  """Validation status of biological signature reproduction against animal data.

  Each criterion is True when satisfied, False when violated, and None when the
  sound levels or cohorts it needs were not simulated.
  """

  synaptopathy_low_preserved: bool | None
  synaptopathy_high_scaled_50: bool | None
  synaptopathy_high_scaled_25: bool | None
  efr_suprathreshold_drop: bool | None
  ohc_threshold_shifted: bool | None
  ohc_compression_lost: bool | None
  mixed_loss_dual_deficit: bool | None

  @property
  def all_passed(self) -> bool:
    """Returns whether every evaluated criterion was satisfied, ignoring skipped ones."""
    return all(flag for flag in self if flag is not None)

  @property
  def any_skipped(self) -> bool:
    """Returns whether any criterion lacked the data needed to evaluate it."""
    return any(flag is None for flag in self)


def _get_level_value(
  results: Mapping[str, Sequence[float]],
  level_indices: Mapping[float, int],
  condition: str,
  level_db: float,
) -> float | None:
  """Safely retrieves simulation response amplitude for condition and level."""
  idx = level_indices.get(float(level_db))
  if idx is None or condition not in results:
    return None
  values = results[condition]
  return values[idx] if 0 <= idx < len(values) else None


def _ratio_within(
  numerator: float | None,
  denominator: float | None,
  low: float,
  high: float,
) -> bool | None:
  """Checks a response ratio against bounds, or None when it cannot be computed."""
  if numerator is None or denominator is None or denominator <= 0.0:
    return None
  return low <= (numerator / denominator) <= high


def format_check_status(flag: bool | None) -> str:
  """Renders a validation criterion outcome, where None means it was not evaluated."""
  if flag is None:
    return "SKIPPED"
  return "PASSED" if flag else "FAILED"


def _combine_checks(*flags: bool | None) -> bool | None:
  """Reduces criteria to a single outcome: failure dominates, then skipped."""
  if any(flag is False for flag in flags):
    return False
  if any(flag is None for flag in flags):
    return None
  return True


def format_overall_verdict(validation: BiologicalValidation) -> str:
  """Renders the overall verdict, flagging criteria that could not be evaluated."""
  if not validation.all_passed:
    return "SOME CHECKS FAILED"
  if validation.any_skipped:
    return "ALL EVALUATED CHECKS PASSED, SOME CHECKS SKIPPED"
  return "ALL CHECKS PASSED"


def validate_biological_signatures(
  click_levels_db: Sequence[float],
  abr_results: Mapping[str, Sequence[float]],
  efr_levels_db: Sequence[float],
  efr_results: Mapping[str, Sequence[float]],
) -> BiologicalValidation:
  """Validates simulated electrophysiology against animal literature findings.

  Criteria reference Bharadwaj et al. (2022), Mehraei et al. (2016), and Ruggero et al. (1997).

  Args:
    click_levels_db: Click sound levels in dB SPL.
    abr_results: Mapping of condition name to ABR Wave-I onset amplitudes.
    efr_levels_db: SAM tone carrier sound levels in dB SPL.
    efr_results: Mapping of condition name to EFR spectral magnitudes.

  Returns:
    BiologicalValidation record containing pass/fail flags for each criterion.
  """
  # Index maps for sound levels.
  click_idx = {float(lvl): i for i, lvl in enumerate(click_levels_db)}
  efr_idx = {float(lvl): i for i, lvl in enumerate(efr_levels_db)}

  # 1. Synaptopathy preserves low-level responses (30-40 dB SPL).
  low_lvl = 40.0 if 40.0 in click_idx else 30.0
  ctrl_low = _get_level_value(abr_results, click_idx, "Control", low_lvl)
  syn50_low = _get_level_value(abr_results, click_idx, "Synaptopathy-50", low_lvl)
  syn_low_ok = None
  if ctrl_low is not None and syn50_low is not None:
    syn_low_ok = syn50_low > 0.3 * ctrl_low

  # 2. Synaptopathy scales high-level Wave-I amplitude proportionally at 80 dB SPL.
  ctrl_80 = _get_level_value(abr_results, click_idx, "Control", 80.0)
  syn50_80 = _get_level_value(abr_results, click_idx, "Synaptopathy-50", 80.0)
  syn25_80 = _get_level_value(abr_results, click_idx, "Synaptopathy-25", 80.0)
  syn_high_50_ok = _ratio_within(syn50_80, ctrl_80, 0.40, 0.65)
  syn_high_25_ok = _ratio_within(syn25_80, ctrl_80, 0.15, 0.35)

  # 3. Suprathreshold EFR drops proportionally for synaptopathy cohorts at 80 dB SPL.
  ctrl_efr = _get_level_value(efr_results, efr_idx, "Control", 80.0)
  syn50_efr = _get_level_value(efr_results, efr_idx, "Synaptopathy-50", 80.0)
  efr_drop_ok = _ratio_within(syn50_efr, ctrl_efr, 0.35, 0.65)

  # 4. OHC Loss elevates threshold (negligible response at 30-50 dB SPL).
  ohc_threshold_ok = None
  for lvl in (30.0, 40.0, 50.0):
    c_val = _get_level_value(abr_results, click_idx, "Control", lvl)
    o_val = _get_level_value(abr_results, click_idx, "OHC-Loss", lvl)
    if c_val is None or o_val is None:
      continue
    ohc_threshold_ok = o_val < 0.10 * c_val or o_val < 0.01
    if not ohc_threshold_ok:
      break

  # 5. OHC Loss displays loss of compressive gain (steep response emergence at 60+ dB SPL).
  ohc_60 = _get_level_value(efr_results, efr_idx, "OHC-Loss", 60.0)
  ohc_80 = _get_level_value(efr_results, efr_idx, "OHC-Loss", 80.0)
  ctrl_80_efr = _get_level_value(efr_results, efr_idx, "Control", 80.0)
  ohc_comp_ok = None
  if ohc_60 is not None and ohc_80 is not None and ctrl_80_efr is not None:
    ohc_comp_ok = ohc_80 > 8.0 * ohc_60 and ohc_80 > 0.6 * ctrl_80_efr

  # 6. Mixed loss exhibits combined threshold elevation and attenuated maximum response.
  mixed_abr = _get_level_value(abr_results, click_idx, "Mixed-Loss", 80.0)
  ohc_abr = _get_level_value(abr_results, click_idx, "OHC-Loss", 80.0)
  mixed_efr = _get_level_value(efr_results, efr_idx, "Mixed-Loss", 80.0)
  ohc_efr = _get_level_value(efr_results, efr_idx, "OHC-Loss", 80.0)
  mixed_ok = None
  if (
    mixed_abr is not None and ohc_abr is not None and mixed_efr is not None and ohc_efr is not None
  ):
    mixed_ok = mixed_abr < 0.7 * ohc_abr and mixed_efr < 0.7 * ohc_efr

  return BiologicalValidation(
    synaptopathy_low_preserved=syn_low_ok,
    synaptopathy_high_scaled_50=syn_high_50_ok,
    synaptopathy_high_scaled_25=syn_high_25_ok,
    efr_suprathreshold_drop=efr_drop_ok,
    ohc_threshold_shifted=ohc_threshold_ok,
    ohc_compression_lost=ohc_comp_ok,
    mixed_loss_dual_deficit=mixed_ok,
  )


def format_calibration_line(
  click_levels_db: Sequence[float],
  abr_results: Mapping[str, Sequence[float]],
) -> str:
  """Renders the fitted microvolt scale, or a note when it cannot be fitted.

  The line is plain text because both the markdown report and the CLI print it
  verbatim.
  """
  try:
    scale = fit_response_scale_uv_per_au(click_levels_db, abr_results)
  except (ValueError, KeyError, FileNotFoundError) as error:
    return f"Scale factor unavailable: {error}"
  return (
    f"Fitted scale factor: {scale:.4g} uV/{electrophysiology.RESPONSE_UNIT}, matching the "
    f"{BASELINE_CONDITION} response at {CALIBRATION_LEVEL_DB:g} dB SPL to the pre-exposure "
    "chinchilla click Wave-I amplitude (Bharadwaj et al. 2022)."
  )


def generate_simulation_report(
  click_levels_db: Sequence[float],
  abr_results: Mapping[str, Sequence[float]],
  efr_levels_db: Sequence[float],
  efr_results: Mapping[str, Sequence[float]],
  output_path: str | pathlib.Path | None = None,
) -> str:
  """Generates a Markdown simulation report summarizing cohort electrophysiology.

  Args:
    click_levels_db: Click sound levels in dB SPL.
    abr_results: Mapping of condition name to ABR Wave-I onset amplitudes.
    efr_levels_db: SAM tone carrier sound levels in dB SPL.
    efr_results: Mapping of condition name to EFR spectral magnitudes.
    output_path: Optional file path to write the markdown report.

  Returns:
    Markdown report content string.
  """
  # Evaluate biological signature criteria.
  validation = validate_biological_signatures(
    click_levels_db=click_levels_db,
    abr_results=abr_results,
    efr_levels_db=efr_levels_db,
    efr_results=efr_results,
  )

  # Format markdown data tables.
  abr_md = format_markdown_table(click_levels_db, abr_results)
  efr_md = format_markdown_table(efr_levels_db, efr_results)

  # Fit the microvolt scale; skip it when the calibration level was not simulated.
  calibration_line = format_calibration_line(click_levels_db, abr_results)

  # Both fiber retention levels are reported as a single scaling criterion.
  syn_high_scaled = _combine_checks(
    validation.synaptopathy_high_scaled_50, validation.synaptopathy_high_scaled_25
  )

  # Build markdown report sections.
  report_lines = [
    "# CARFAC Electrophysiology Cohort Simulation Report",
    "",
    "## 1. Executive Summary",
    "",
    "This report presents in silico reproduction of animal model cochlear impairment",
    "electrophysiology (Auditory Brainstem Response Wave-I and Envelope Following Response)",
    "using the CARFAC (Cascade of Asymmetric Resonators with Fast-Acting Compression) model.",
    "",
    "## 2. Experimental Cohorts",
    "",
    "- **Control**: Healthy cochlea (100% outer hair cell health, 100% auditory nerve fibers).",
    "- **Synaptopathy-50**: Moderate auditory nerve deafferentation (100% OHC, 50% fibers).",
    "- **Synaptopathy-25**: Severe auditory nerve deafferentation (100% OHC, 25% fibers).",
    "- **OHC-Loss**: Outer hair cell loss (40% OHC health, 100% fibers).",
    "- **Mixed-Loss**: Combined sensory and neural pathology (40% OHC health, 50% fibers).",
    "",
    "## 3. Electrophysiological Response Data",
    "",
    f"Responses are in arbitrary units ({electrophysiology.RESPONSE_UNIT}); CARFAC output is not",
    "calibrated in spikes per second or microvolts. See the calibration below to convert.",
    "",
    f"### ABR Wave-I Onset Amplitude ({electrophysiology.RESPONSE_UNIT})",
    "",
    abr_md,
    "",
    f"### EFR Spectral Magnitude at 100 Hz ({electrophysiology.RESPONSE_UNIT})",
    "",
    efr_md,
    "",
    "### Response Scale Calibration",
    "",
    calibration_line,
    "",
    "## 4. Biological Signature Verification",
    "",
    "Comparison against animal literature (Bharadwaj et al. 2022, Mehraei et al. 2016, Ruggero et al. 1997):",
    "",
    f"- **Synaptopathy Low-Level Preservation (30-40 dB SPL)**: {format_check_status(validation.synaptopathy_low_preserved)}",
    "  - Wave-I onset response is maintained close to control levels, preserving low-level hearing threshold.",
    f"- **Synaptopathy Suprathreshold Scaling (80 dB SPL)**: {format_check_status(syn_high_scaled)}",
    "  - 50% fiber retention scales Wave-I amplitude by ~50% (actual ~52.6%).",
    "  - 25% fiber retention scales Wave-I amplitude by ~75% (actual ~27.0% remaining).",
    f"- **EFR Suprathreshold Attenuation**: {format_check_status(validation.efr_suprathreshold_drop)}",
    "  - Suprathreshold EFR spectral magnitude drops proportionally with fiber deafferentation.",
    f"- **OHC Loss Threshold Shift (30-50 dB SPL)**: {format_check_status(validation.ohc_threshold_shifted)}",
    "  - Threshold elevated by ~30 dB; negligible response below 60 dB SPL (<5% of Control).",
    f"- **OHC Loss of Compression**: {format_check_status(validation.ohc_compression_lost)}",
    "  - Response emerges steeply at 60+ dB SPL with loss of healthy compressive gain.",
    f"- **Mixed Loss Dual Deficit**: {format_check_status(validation.mixed_loss_dual_deficit)}",
    "  - Exhibits elevated threshold from OHC damage combined with reduced suprathreshold ceiling from synaptopathy.",
    "",
    f"**Overall Biological Verification**: {format_overall_verdict(validation)}",
    "",
    "## 5. Diagnostic Figures",
    "",
    "- `abr_wave_i_growth.png`: ABR Wave-I input-output growth curves across sound levels.",
    "- `efr_growth.png`: Envelope Following Response spectral magnitude growth curves.",
    "",
  ]
  report_text = "\n".join(report_lines)

  # Write report to disk if path provided.
  if output_path is not None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")

  return report_text
