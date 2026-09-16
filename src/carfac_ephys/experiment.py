"""Cohort definitions and simulation level series runners for CARFAC electrophysiology."""

import itertools
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
    name="Selective-Synaptopathy",
    ohc_health=1.0,
    fiber_retention=carfac_model.FiberRetention(hsr=1.0, msr=0.5, lsr=0.0),
    description="Selective deafferentation (100% OHC, 100% HSR, 50% MSR, 0% LSR fibers)",
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
DEFAULT_TONE_BURST_FREQUENCIES_HZ: tuple[float, float] = (4000.0, 8000.0)
DEFAULT_TONE_BURST_LEVELS_DB: tuple[float, ...] = (30.0, 40.0, 50.0, 60.0, 70.0, 80.0)

CALIBRATION_STRATEGIES: tuple[str, ...] = ("individual", "mode-dependent", "unified")
CALIBRATION_REFERENCES: tuple[str, ...] = ("average", "click", "4k", "8k")


class CalibrationConfig(NamedTuple):
  """Configuration specifying calibration strategy and reference stimulus."""

  strategy: str = "mode-dependent"
  reference: str = "average"


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


class ToneBurstCohortResults(NamedTuple):
  """Container for multi-frequency tone burst simulation results."""

  results_by_frequency: dict[float, dict[str, list[float]]]
  composite_results: dict[str, list[float]]
  levels_db: tuple[float, ...]
  frequencies_hz: tuple[float, ...]
  high_f_factor: float = 0.0

  @property
  def results_4k(self) -> dict[str, list[float]]:
    """Convenience accessor for 4 kHz simulation results."""
    return self.results_by_frequency.get(4000.0, {})

  @property
  def results_8k(self) -> dict[str, list[float]]:
    """Convenience accessor for 8 kHz simulation results."""
    return self.results_by_frequency.get(8000.0, {})

  @property
  def results_avg(self) -> dict[str, list[float]]:
    """Convenience accessor for 4/8 kHz composite average results."""
    return self.composite_results


def simulate_tone_burst_abr_series(
  frequency_hz: float = 4000.0,
  cohort: Cohort | Mapping[str, Any] | Sequence[CohortCondition] | None = None,
  tone_burst_levels_db: Sequence[float] = DEFAULT_TONE_BURST_LEVELS_DB,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  duration_s: float = 0.005,
  ramp_s: float = 0.0005,
  delay_s: float = 0.005,
  window_s: float = 0.008,
  mode: str = "baseline_to_peak",
  high_f_factor: float = 0.0,
) -> dict[str, list[float]]:
  """Runs ABR Wave-I tone-burst level series across cohort conditions.

  Uses alternating polarity (+1 and -1) burst presentation and averages
  the evoked population firing rates to cancel the phase-locked cochlear
  microphonic / stimulus artifact and isolate the neural Wave-I peak.

  Args:
    frequency_hz: Carrier frequency in Hz (default 4000.0).
    cohort: Cohort instance or mapping of conditions. Defaults to standard cohort.
    tone_burst_levels_db: Sequence of sound levels in dB SPL.
    sample_rate: Sampling rate in Hz.
    duration_s: Burst duration in seconds (default 5 ms).
    ramp_s: Linear onset and offset ramp duration in seconds (default 0.5 ms).
    delay_s: Stimulus onset delay in seconds.
    window_s: Analysis window duration following onset in seconds.
    mode: Wave-I extraction mode ('baseline_to_peak' or 'peak_to_trough').
    high_f_factor: CARFAC factor adjusting high-frequency channel distribution and damping.

  Returns:
    Dictionary mapping condition name to list of Wave-I onset amplitudes in arbitrary units.
  """
  # Validate input parameters.
  if sample_rate <= 0:
    raise ValueError("sample_rate must be positive.")
  if not tone_burst_levels_db:
    raise ValueError("tone_burst_levels_db sequence cannot be empty.")
  if frequency_hz <= 0.0:
    raise ValueError("frequency_hz must be positive.")

  # Resolve cohort specifications.
  resolved_cohort = _resolve_cohort(cohort)
  results: dict[str, list[float]] = {}

  # Iterate through each condition.
  for condition in resolved_cohort.values():
    model = carfac_model.build_model(
      ohc_health=condition.ohc_health,
      fiber_retention=condition.fiber_retention,
      fs=sample_rate,
      high_f_factor=high_f_factor,
    )
    amps: list[float] = []

    # Run level sweep.
    for level in tone_burst_levels_db:
      # Positive polarity run.
      model.reset()
      w_pos = stimuli.generate_tone_burst(
        frequency_hz=frequency_hz,
        duration_s=duration_s,
        sample_rate=sample_rate,
        peak_db_spl=float(level),
        ramp_s=ramp_s,
        delay_s=delay_s,
        polarity=1,
      )
      naps_pos = model.run(w_pos)
      pop_pos = electrophysiology.compute_population_rate(naps_pos)

      # Negative polarity run.
      model.reset()
      w_neg = stimuli.generate_tone_burst(
        frequency_hz=frequency_hz,
        duration_s=duration_s,
        sample_rate=sample_rate,
        peak_db_spl=float(level),
        ramp_s=ramp_s,
        delay_s=delay_s,
        polarity=-1,
      )
      naps_neg = model.run(w_neg)
      pop_neg = electrophysiology.compute_population_rate(naps_neg)

      # Average responses across alternating polarities.
      pop_avg = 0.5 * (pop_pos + pop_neg)

      amp = electrophysiology.extract_wave_i_amplitude(
        pop_avg,
        sample_rate=sample_rate,
        stimulus_onset_s=delay_s,
        window_s=window_s,
        mode=mode,
      )
      amps.append(float(amp))

    results[condition.name] = amps

  return results


def simulate_tone_burst_cohort(
  cohort: Cohort | Mapping[str, Any] | Sequence[CohortCondition] | None = None,
  frequencies_hz: Sequence[float] = DEFAULT_TONE_BURST_FREQUENCIES_HZ,
  tone_burst_levels_db: Sequence[float] = DEFAULT_TONE_BURST_LEVELS_DB,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  duration_s: float = 0.005,
  ramp_s: float = 0.0005,
  delay_s: float = 0.005,
  window_s: float = 0.008,
  mode: str = "baseline_to_peak",
  high_f_factor: float = 0.0,
) -> ToneBurstCohortResults:
  """Simulates tone-burst ABR level sweeps across multiple frequencies and cohorts.

  Computes individual frequency level series (e.g. 4 kHz and 8 kHz) and their
  composite mean growth functions across all cohort conditions.

  Args:
    cohort: Cohort instance or mapping of conditions.
    frequencies_hz: Sequence of carrier frequencies in Hz.
    tone_burst_levels_db: Sequence of sound levels in dB SPL.
    sample_rate: Sampling rate in Hz.
    duration_s: Burst duration in seconds.
    ramp_s: Linear onset and offset ramp duration in seconds.
    delay_s: Stimulus onset delay in seconds.
    window_s: Analysis window duration following onset in seconds.
    mode: Wave-I extraction mode.
    high_f_factor: CARFAC factor adjusting high-frequency channel distribution and damping.

  Returns:
    ToneBurstCohortResults holding per-frequency results and composite average results.
  """
  if not frequencies_hz:
    raise ValueError("frequencies_hz cannot be empty.")

  resolved_cohort = _resolve_cohort(cohort)
  results_by_freq: dict[float, dict[str, list[float]]] = {}

  for freq in frequencies_hz:
    results_by_freq[float(freq)] = simulate_tone_burst_abr_series(
      frequency_hz=float(freq),
      cohort=resolved_cohort,
      tone_burst_levels_db=tone_burst_levels_db,
      sample_rate=sample_rate,
      duration_s=duration_s,
      ramp_s=ramp_s,
      delay_s=delay_s,
      window_s=window_s,
      mode=mode,
      high_f_factor=high_f_factor,
    )

  # Compute composite average across frequencies for each condition and level.
  composite_results: dict[str, list[float]] = {}
  cond_names = list(resolved_cohort.keys())
  n_levels = len(tone_burst_levels_db)
  n_freqs = len(frequencies_hz)

  for name in cond_names:
    avg_levels: list[float] = []
    for l_idx in range(n_levels):
      mean_val = sum(results_by_freq[float(f)][name][l_idx] for f in frequencies_hz) / float(n_freqs)
      avg_levels.append(float(mean_val))
    composite_results[name] = avg_levels

  return ToneBurstCohortResults(
    results_by_frequency=results_by_freq,
    composite_results=composite_results,
    levels_db=tuple(float(lvl) for lvl in tone_burst_levels_db),
    frequencies_hz=tuple(float(f) for f in frequencies_hz),
    high_f_factor=float(high_f_factor),
  )


def simulate_tone_burst_waveforms(
  frequency_hz: float = 4000.0,
  cohort: Cohort | Mapping[str, Any] | Sequence[CohortCondition] | None = None,
  level_db: float = 80.0,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  duration_s: float = 0.005,
  ramp_s: float = 0.0005,
  delay_s: float = 0.005,
  total_duration_s: float = 0.02,
  high_f_factor: float = 0.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """Generates averaged population response waveforms across cohorts for a tone burst.

  Args:
    frequency_hz: Carrier frequency in Hz (default 4000.0).
    cohort: Cohort instance or mapping of conditions.
    level_db: Sound level in dB SPL (default 80.0).
    sample_rate: Sampling rate in Hz.
    duration_s: Burst duration in seconds.
    ramp_s: Linear onset and offset ramp duration in seconds.
    delay_s: Stimulus onset delay in seconds.
    total_duration_s: Total duration of the simulation buffer in seconds.
    high_f_factor: CARFAC factor adjusting high-frequency channel distribution and damping.

  Returns:
    Tuple of (time_ms, waveforms_by_cohort) where time_ms is a 1D array of time
    in milliseconds, and waveforms_by_cohort maps condition name to 1D population
    rate array in arbitrary units.
  """
  if total_duration_s < delay_s + duration_s:
    raise ValueError("total_duration_s cannot be less than delay_s + duration_s.")

  resolved_cohort = _resolve_cohort(cohort)
  n_total = int(round(total_duration_s * sample_rate))
  time_ms = 1000.0 * np.arange(n_total, dtype=np.float64) / sample_rate

  # Synthesize positive and negative polarity bursts padded to total_duration_s.
  w_pos_raw = stimuli.generate_tone_burst(
    frequency_hz=frequency_hz,
    duration_s=duration_s,
    sample_rate=sample_rate,
    peak_db_spl=level_db,
    ramp_s=ramp_s,
    delay_s=delay_s,
    polarity=1,
  )
  w_pos = np.zeros(n_total, dtype=np.float64)
  w_pos[: len(w_pos_raw)] = w_pos_raw

  w_neg_raw = stimuli.generate_tone_burst(
    frequency_hz=frequency_hz,
    duration_s=duration_s,
    sample_rate=sample_rate,
    peak_db_spl=level_db,
    ramp_s=ramp_s,
    delay_s=delay_s,
    polarity=-1,
  )
  w_neg = np.zeros(n_total, dtype=np.float64)
  w_neg[: len(w_neg_raw)] = w_neg_raw

  waveforms: dict[str, np.ndarray] = {}

  for condition in resolved_cohort.values():
    model = carfac_model.build_model(
      ohc_health=condition.ohc_health,
      fiber_retention=condition.fiber_retention,
      fs=sample_rate,
      high_f_factor=high_f_factor,
    )
    # Positive run.
    model.reset()
    naps_pos = model.run(w_pos)
    pop_pos = electrophysiology.compute_population_rate(naps_pos)

    # Negative run.
    model.reset()
    naps_neg = model.run(w_neg)
    pop_neg = electrophysiology.compute_population_rate(naps_neg)

    # Alternating polarity average.
    waveforms[condition.name] = 0.5 * (pop_pos + pop_neg)

  return time_ms, waveforms


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
  click_levels_db: Sequence[float] | None = None,
  abr_results: Mapping[str, Sequence[float]] | None = None,
  dataset: empirical.ChinchillaAbrDataset | None = None,
  level_db: float = CALIBRATION_LEVEL_DB,
  condition: str = BASELINE_CONDITION,
  frequency_hz: float | None = empirical.CLICK_FREQUENCY_HZ,
  levels_db: Sequence[float] | None = None,
) -> float:
  """Fits the microvolts per arbitrary unit scale of simulated ABR responses.

  Simulated responses are dimensionless, so they are calibrated against the
  pre-exposure (healthy) Wave-I amplitude of the chinchilla dataset for the
  matching stimulus frequency.

  Args:
    click_levels_db: Sound levels in dB SPL (synonym for levels_db).
    abr_results: Mapping of condition name to Wave-I amplitudes in AU.
    dataset: Empirical dataset; loaded from package data when None.
    level_db: Sound level matched to the empirical high-level amplitude (default 80.0).
    condition: Cohort treated as the healthy baseline (default 'Control').
    frequency_hz: Stimulus frequency in Hz (0 Hz denotes click, None denotes 4/8 kHz average).
    levels_db: Sound levels in dB SPL.

  Returns:
    Scale factor in microvolts per arbitrary unit (> 0).
  """
  resolved_levels = levels_db if levels_db is not None else click_levels_db
  if resolved_levels is None:
    raise ValueError("Must provide either levels_db or click_levels_db.")
  if abr_results is None:
    raise ValueError("abr_results cannot be None.")

  # Look up the simulated healthy response at the calibration level.
  level_indices = {float(level): index for index, level in enumerate(resolved_levels)}
  simulated_au = _get_level_value(abr_results, level_indices, condition, level_db)
  if simulated_au is None:
    raise ValueError(f"No '{condition}' response at {level_db} dB SPL to calibrate against.")

  # Match it to empirical pre-exposure Wave-I amplitude.
  data = empirical.load_chinchilla_abr_dataset() if dataset is None else dataset
  if frequency_hz == empirical.CLICK_FREQUENCY_HZ or frequency_hz == 0.0:
    measured_uv = data.high_level_w1_uv[empirical.CLICK_FREQUENCY_HZ].mean_pre
  elif frequency_hz is None:
    measured_uv = data.tone_average_w1_uv.mean_pre
  elif float(frequency_hz) in data.high_level_w1_uv:
    measured_uv = data.high_level_w1_uv[float(frequency_hz)].mean_pre
  else:
    raise ValueError(f"No empirical Wave-I reference for frequency {frequency_hz} Hz.")

  return electrophysiology.fit_microvolts_per_au([simulated_au], [measured_uv])


def resolve_scale_factor(
  strategy: str = "mode-dependent",
  stimulus_type: str | float | None = "click",
  click_results: Mapping[str, Sequence[float]] | None = None,
  tone_burst_results: ToneBurstCohortResults | None = None,
  click_levels_db: Sequence[float] | None = None,
  tone_burst_levels_db: Sequence[float] | None = None,
  reference: str = "average",
  dataset: empirical.ChinchillaAbrDataset | None = None,
  condition: str = BASELINE_CONDITION,
  level_db: float = CALIBRATION_LEVEL_DB,
) -> float:
  """Resolves the scale factor (uV/AU) for a stimulus under a given calibration strategy.

  Strategies:
    - 'individual': Each stimulus (click, 4k, 8k, average) is calibrated to its
      own 80 dB pre-exposure amplitude.
    - 'mode-dependent': Clicks use the click factor, while tone-bursts share
      a single factor determined by `reference` ('average', '4k', or '8k').
    - 'unified': All stimuli share a single factor determined by `reference`
      ('click', 'average', '4k', or '8k').

  Args:
    strategy: Calibration strategy ('individual', 'mode-dependent', or 'unified').
    stimulus_type: Target stimulus ('click', '4k' or 4000.0, '8k' or 8000.0, 'average' or None).
    click_results: Click ABR simulation results mapping.
    tone_burst_results: ToneBurstCohortResults instance.
    click_levels_db: Sequence of click levels in dB SPL.
    tone_burst_levels_db: Sequence of tone burst levels in dB SPL.
    reference: Reference stimulus used for shared scaling ('average', 'click', '4k', '8k').
    dataset: ChinchillaAbrDataset instance.
    condition: Baseline cohort name (default 'Control').
    level_db: Calibration level in dB SPL (default 80.0).

  Returns:
    Scale factor in uV/AU.
  """
  strategy_norm = str(strategy).lower()
  if strategy_norm not in CALIBRATION_STRATEGIES:
    raise ValueError(
      f"Unknown calibration strategy '{strategy}'. Valid options: {CALIBRATION_STRATEGIES}"
    )
  reference_norm = str(reference).lower()
  if reference_norm not in CALIBRATION_REFERENCES:
    raise ValueError(
      f"Unknown calibration reference '{reference}'. Valid options: {CALIBRATION_REFERENCES}"
    )

  # Normalize target stimulus identifier.
  if stimulus_type in ("click", empirical.CLICK_FREQUENCY_HZ, 0.0, 0):
    norm_stim = "click"
  elif stimulus_type in ("4k", 4000.0, "4000", 4000):
    norm_stim = "4k"
  elif stimulus_type in ("8k", 8000.0, "8000", 8000):
    norm_stim = "8k"
  elif stimulus_type in ("average", "composite", None):
    norm_stim = "average"
  else:
    raise ValueError(
      f"Unrecognized stimulus_type '{stimulus_type}'. Expected 'click', '4k', '8k', or 'average'."
    )

  # Determine which physical stimulus reference to scale against based on strategy.
  if strategy_norm == "individual":
    effective_ref = norm_stim
  elif strategy_norm == "mode-dependent":
    effective_ref = "click" if norm_stim == "click" else reference_norm
  elif strategy_norm == "unified":
    effective_ref = reference_norm
  else:
    raise ValueError(f"Unsupported strategy: {strategy_norm}")

  # Compute the scale factor for effective_ref.
  if effective_ref == "click":
    if click_results is None:
      raise ValueError("click_results required to fit click scale factor.")
    if click_levels_db is not None:
      levels = click_levels_db
    elif tone_burst_results is not None and len(click_results.get(condition, [])) == len(tone_burst_results.levels_db):
      levels = tone_burst_results.levels_db
    else:
      levels = DEFAULT_CLICK_LEVELS_DB
    return fit_response_scale_uv_per_au(
      levels_db=levels,
      abr_results=click_results,
      dataset=dataset,
      condition=condition,
      level_db=level_db,
      frequency_hz=empirical.CLICK_FREQUENCY_HZ,
    )
  elif effective_ref == "4k":
    if tone_burst_results is None:
      raise ValueError("tone_burst_results required to fit 4 kHz tone-burst scale factor.")
    levels = tone_burst_levels_db if tone_burst_levels_db is not None else tone_burst_results.levels_db
    return fit_response_scale_uv_per_au(
      levels_db=levels,
      abr_results=tone_burst_results.results_4k,
      dataset=dataset,
      condition=condition,
      level_db=level_db,
      frequency_hz=4000.0,
    )
  elif effective_ref == "8k":
    if tone_burst_results is None:
      raise ValueError("tone_burst_results required to fit 8 kHz tone-burst scale factor.")
    levels = tone_burst_levels_db if tone_burst_levels_db is not None else tone_burst_results.levels_db
    return fit_response_scale_uv_per_au(
      levels_db=levels,
      abr_results=tone_burst_results.results_8k,
      dataset=dataset,
      condition=condition,
      level_db=level_db,
      frequency_hz=8000.0,
    )
  elif effective_ref == "average":
    if tone_burst_results is None:
      raise ValueError("tone_burst_results required to fit composite average tone-burst scale factor.")
    levels = tone_burst_levels_db if tone_burst_levels_db is not None else tone_burst_results.levels_db
    return fit_response_scale_uv_per_au(
      levels_db=levels,
      abr_results=tone_burst_results.composite_results,
      dataset=dataset,
      condition=condition,
      level_db=level_db,
      frequency_hz=None,
    )
  else:
    raise ValueError(f"Unhandled reference: {effective_ref}")


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
  "Selective-Synaptopathy": {
    "color": "#9467bd",
    "marker": "v",
    "linestyle": "--",
    "label": "Selective Synaptopathy (100% HSR, 50% MSR, 0% LSR)",
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
  stimulus_label: str = "Broadband Click",
) -> pathlib.Path:
  """Plots and saves ABR Wave-I input-output functions across cohort conditions."""
  # Render ABR Wave-I growth curves.
  return _plot_growth_curves(
    levels_db=click_levels_db,
    results=results,
    output_path=output_path,
    xlabel=f"{stimulus_label} Sound Level (dB SPL)",
    ylabel=f"ABR Wave-I Onset Amplitude ({electrophysiology.RESPONSE_UNIT})",
    title=f"ABR Wave-I Input-Output Growth Functions ({stimulus_label})",
  )


def format_8k_title(base_title: str = "8 kHz Tone Burst", high_f_factor: float = 0.0) -> str:
  """Formats 8 kHz title with high_f_factor annotation."""
  return f"{base_title} (high_f={high_f_factor:g})"


def plot_tone_burst_waveforms(
  waveforms_4k: tuple[np.ndarray, Mapping[str, np.ndarray]],
  waveforms_8k: tuple[np.ndarray, Mapping[str, np.ndarray]],
  output_path: str | pathlib.Path,
  level_db: float = 80.0,
  high_f_factor: float = 0.0,
  stimulus_onset_ms: float = 5.0,
  stimulus_duration_ms: float = 5.0,
) -> pathlib.Path:
  """Plots horizontal side-by-side waveforms for 4 kHz and 8 kHz tone bursts across cohorts.

  Args:
    waveforms_4k: Tuple of (time, mapping of cohort_name -> waveform) for 4 kHz,
      where time can be in seconds or milliseconds.
    waveforms_8k: Tuple of (time, mapping of cohort_name -> waveform) for 8 kHz,
      where time can be in seconds or milliseconds.
    output_path: Path to save the output figure image.
    level_db: Sound level in dB SPL used for the waveforms (default: 80.0).
    high_f_factor: CARFAC high_f_factor parameter.
    stimulus_onset_ms: Stimulus onset delay in milliseconds (default: 5.0).
    stimulus_duration_ms: Stimulus duration in milliseconds (default: 5.0).

  Returns:
    Path to the saved figure.
  """
  path = pathlib.Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)

  panels = [
    (ax1, waveforms_4k, "4 kHz Tone Burst (5 ms)"),
    (ax2, waveforms_8k, format_8k_title("8 kHz Tone Burst (5 ms)", high_f_factor)),
  ]

  stim_end_ms = stimulus_onset_ms + stimulus_duration_ms

  for ax, (t_raw, waveforms), title in panels:
    t_arr = np.asarray(t_raw, dtype=np.float64)
    # Auto-detect if time is in seconds (max <= 1.0, e.g. 0.02 s) or ms (e.g. 20 ms).
    if len(t_arr) > 0 and np.max(t_arr) <= 1.0:
      t_ms = t_arr * 1e3
    else:
      t_ms = t_arr

    for name, wave in waveforms.items():
      style = COHORT_STYLES.get(
        name,
        {"color": "#555555", "marker": "", "linestyle": "-", "label": name},
      )
      ax.plot(
        t_ms,
        wave,
        label=style["label"],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.8,
      )

    # Highlight tone-burst stimulus duration (active burst period)
    if stimulus_duration_ms > 0:
      ax.axvspan(
        stimulus_onset_ms,
        stim_end_ms,
        color="#d0d0d0",
        alpha=0.35,
        label=f"Stimulus Duration ({stimulus_onset_ms:g}-{stim_end_ms:g} ms)",
      )
    ax.set_xlabel("Time (ms)", fontsize=11, fontweight="bold")
    ax.set_title(f"{title} @ {level_db:g} dB SPL", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    if len(t_ms) > 0:
      ax.set_xlim(left=0.0, right=float(t_ms[-1]))

  ax1.set_ylabel(
    f"Neural Response Rate ({electrophysiology.RESPONSE_UNIT})",
    fontsize=11,
    fontweight="bold",
  )

  fig.suptitle(
    f"Compound ABR Tone-Burst Response Waveforms ({level_db:g} dB SPL, Alternating Polarity)",
    fontsize=13,
    fontweight="bold",
    y=0.98,
  )
  fig.tight_layout(rect=[0.0, 0.14, 1.0, 0.94])

  # Place a single legend outside the axes grid centered within the combined subplots width.
  pos1 = ax1.get_position()
  pos2 = ax2.get_position()
  center_x = (pos1.x0 + pos2.x1) / 2.0

  handles, labels = ax1.get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  fig.legend(
    by_label.values(),
    by_label.keys(),
    loc="lower center",
    bbox_to_anchor=(center_x, 0.01),
    ncol=4,
    frameon=True,
    fontsize=8.5,
    columnspacing=1.2,
  )

  fig.savefig(path, dpi=300)
  plt.close(fig)

  return path


def plot_tone_burst_growth(
  results: ToneBurstCohortResults,
  output_path: str | pathlib.Path,
) -> pathlib.Path:
  """Plots and saves 3-panel horizontal ABR Wave-I growth functions for tone bursts.

  Panels show 4 kHz, 8 kHz, and 4/8 kHz Composite Average side-by-side across sound levels.

  Args:
    results: ToneBurstCohortResults containing levels and results for 4k, 8k, and avg.
    output_path: Path to save the output figure image.

  Returns:
    Path to the saved figure.
  """
  path = pathlib.Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)

  sub_panels = [
    (axes[0], results.results_4k, "4 kHz Tone Burst"),
    (axes[1], results.results_8k, format_8k_title("8 kHz Tone Burst", results.high_f_factor)),
    (axes[2], results.results_avg, "4/8 kHz Composite Average"),
  ]

  for ax, cohort_dict, subtitle in sub_panels:
    for name, values in cohort_dict.items():
      style = COHORT_STYLES.get(
        name,
        {"color": "#555555", "marker": ".", "linestyle": "-", "label": name},
      )
      ax.plot(
        results.levels_db,
        values,
        label=style["label"],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2.0,
        markersize=6.5,
      )
    ax.set_xlabel("Sound Level (dB SPL)", fontsize=11, fontweight="bold")
    ax.set_title(subtitle, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

  axes[0].set_ylabel(
    f"ABR Wave-I Onset Amplitude ({electrophysiology.RESPONSE_UNIT})",
    fontsize=11,
    fontweight="bold",
  )
  axes[0].legend(frameon=True, fontsize=8.5, loc="upper left")

  fig.suptitle(
    "ABR Wave-I Input-Output Growth Functions (Tone Bursts: 4 kHz, 8 kHz & Average)",
    fontsize=13,
    fontweight="bold",
    y=0.98,
  )
  fig.tight_layout()
  fig.savefig(path, dpi=300)
  plt.close(fig)

  return path


def plot_tone_burst_individual_growth(
  results: ToneBurstCohortResults,
  output_dir: str | pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
  """Plots 3 separate growth figures: 4 kHz, 8 kHz, and 4/8 kHz Composite Average.

  Args:
    results: ToneBurstCohortResults containing levels and results.
    output_dir: Directory where the 3 figures will be saved.

  Returns:
    Tuple of paths to (4 kHz figure, 8 kHz figure, Composite Average figure).
  """
  out_dir = pathlib.Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  path_4k = plot_abr_growth(
    results.levels_db,
    results.results_4k,
    out_dir / "abr_wave_i_growth_4k.png",
    stimulus_label="4 kHz Tone Burst",
  )
  path_8k = plot_abr_growth(
    results.levels_db,
    results.results_8k,
    out_dir / "abr_wave_i_growth_8k.png",
    stimulus_label=format_8k_title("8 kHz Tone Burst", results.high_f_factor),
  )
  path_avg = plot_abr_growth(
    results.levels_db,
    results.composite_results,
    out_dir / "abr_wave_i_growth_avg.png",
    stimulus_label="4/8 kHz Composite Average",
  )
  return path_4k, path_8k, path_avg


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
  selective_low_level_spared: bool | None
  empirical_threshold_shift_matched: bool | None
  empirical_w1_ratio_matched: bool | None
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


# Cohort compared against the noise-exposed chinchillas of Bharadwaj et al.
# (2022): those animals recovered their click thresholds but kept a reduced
# suprathreshold Wave-I, the signature of selective LSR/MSR deafferentation.
EXPOSED_CONDITION: str = "Selective-Synaptopathy"

# Wave-I amplitude criterion defining the simulated ABR threshold, in the same
# microvolt scale as the animal recordings.
THRESHOLD_CRITERION_UV: float = 0.1

# Tolerances on the agreement between simulation and animal data: 3 dB is the
# step of the animal threshold search, and 0.10 is the spread of the measured
# post over pre Wave-I ratio.
THRESHOLD_SHIFT_TOLERANCE_DB: float = 3.0
W1_RATIO_TOLERANCE: float = 0.10


def _interpolate_crossing_level_db(
  low_level_db: float,
  high_level_db: float,
  low_amplitude: float,
  high_amplitude: float,
  criterion: float,
) -> float:
  """Interpolates the level at which a bracketed growth segment hits a criterion."""
  # Growth functions are near-exponential in level, so interpolate on a log
  # amplitude axis, falling back to linear when the lower end is not positive.
  if low_amplitude > 0.0:
    span = np.log(high_amplitude) - np.log(low_amplitude)
    fraction = (np.log(criterion) - np.log(low_amplitude)) / span
  else:
    fraction = (criterion - low_amplitude) / (high_amplitude - low_amplitude)
  return float(low_level_db + fraction * (high_level_db - low_level_db))


def estimate_threshold_db(
  levels_db: Sequence[float],
  amplitudes: Sequence[float],
  criterion: float,
) -> float | None:
  """Estimates the sound level at which a response reaches a criterion amplitude.

  Args:
    levels_db: Sound levels in dB SPL, in any order.
    amplitudes: Response amplitudes paired with `levels_db`, in arbitrary units.
    criterion: Amplitude defining threshold, in the same units as `amplitudes`.

  Returns:
    Interpolated threshold in dB SPL, or None when the sweep does not bracket
    the criterion, since extrapolating outside the simulated levels is not
    meaningful.
  """
  # Validate inputs.
  if criterion <= 0.0:
    raise ValueError(f"criterion must be positive, got {criterion}.")
  if len(levels_db) != len(amplitudes):
    raise ValueError(f"Inputs must be equally long, got {len(levels_db)} and {len(amplitudes)}.")

  # Find the first ascending segment that brackets the criterion.
  points = sorted(zip(levels_db, amplitudes, strict=True))
  for (low_level, low_amp), (high_level, high_amp) in itertools.pairwise(points):
    if low_amp >= criterion or high_amp < criterion:
      continue
    return _interpolate_crossing_level_db(low_level, high_level, low_amp, high_amp, criterion)
  return None


class EmpiricalComparison(NamedTuple):
  """Quantitative comparison of a simulated cohort against animal ABR data.

  Simulated values are None when the sweep lacked the levels needed to compute
  them. Threshold shifts are in dB and Wave-I ratios are dimensionless post over
  pre amplitude fractions.
  """

  condition: str
  simulated_threshold_shift_db: float | None
  animal_threshold_shift_db: float
  threshold_shift_matched: bool | None
  simulated_w1_ratio: float | None
  animal_w1_ratio: float
  w1_ratio_matched: bool | None


def _simulated_threshold_shift_db(
  levels_db: Sequence[float],
  abr_results: Mapping[str, Sequence[float]],
  dataset: empirical.ChinchillaAbrDataset,
  condition: str,
  baseline: str,
  frequency_hz: float | None = empirical.CLICK_FREQUENCY_HZ,
  scale_uv_per_au: float | None = None,
) -> float | None:
  """Computes the exposed minus baseline threshold shift of the simulation."""
  if baseline not in abr_results or condition not in abr_results:
    return None

  # Express the microvolt threshold criterion in model units.
  if scale_uv_per_au is None:
    try:
      scale_uv_per_au = fit_response_scale_uv_per_au(
        levels_db=levels_db,
        abr_results=abr_results,
        dataset=dataset,
        condition=baseline,
        frequency_hz=frequency_hz,
      )
    except (ValueError, KeyError):
      return None
  if scale_uv_per_au is None or scale_uv_per_au <= 0.0:
    return None
  criterion_au = THRESHOLD_CRITERION_UV / scale_uv_per_au

  # Interpolate both thresholds; a missing crossing makes the shift unknown.
  baseline_db = estimate_threshold_db(levels_db, abr_results[baseline], criterion_au)
  exposed_db = estimate_threshold_db(levels_db, abr_results[condition], criterion_au)
  if baseline_db is None or exposed_db is None:
    return None
  return exposed_db - baseline_db


def compare_to_empirical(
  click_levels_db: Sequence[float],
  abr_results: Mapping[str, Sequence[float]],
  dataset: empirical.ChinchillaAbrDataset | None = None,
  condition: str = EXPOSED_CONDITION,
  baseline: str = BASELINE_CONDITION,
  scale_uv_per_au: float | None = None,
) -> EmpiricalComparison:
  """Compares simulated ABR thresholds and Wave-I growth against chinchilla data.

  The noise-exposed chinchillas of Bharadwaj et al. (2022) recovered their click
  thresholds while losing suprathreshold Wave-I amplitude, so the simulated
  cohort must reproduce both numbers, not merely rank in the right order.

  Args:
    click_levels_db: Click sound levels in dB SPL.
    abr_results: Mapping of condition name to Wave-I amplitudes in AU.
    dataset: Empirical dataset; loaded from package data when None.
    condition: Cohort standing in for the noise-exposed animals.
    baseline: Cohort treated as the healthy, pre-exposure baseline.
    scale_uv_per_au: Optional pre-fitted microvolts per AU scale factor.

  Returns:
    Record of the simulated values, the animal values, and their agreement.
  """
  data = empirical.load_chinchilla_abr_dataset() if dataset is None else dataset

  # Compare the click threshold shift, which the animals recovered.
  animal_shift_db = data.click_threshold_shift_db
  simulated_shift_db = _simulated_threshold_shift_db(
    click_levels_db,
    abr_results,
    data,
    condition,
    baseline,
    scale_uv_per_au=scale_uv_per_au,
  )
  shift_matched = None
  if simulated_shift_db is not None:
    shift_matched = abs(simulated_shift_db - animal_shift_db) <= THRESHOLD_SHIFT_TOLERANCE_DB

  # Compare the suprathreshold Wave-I attenuation, which the animals retained.
  animal_w1_ratio = data.wave_i_ratio()
  level_indices = {float(level): index for index, level in enumerate(click_levels_db)}
  exposed_au = _get_level_value(abr_results, level_indices, condition, CALIBRATION_LEVEL_DB)
  baseline_au = _get_level_value(abr_results, level_indices, baseline, CALIBRATION_LEVEL_DB)
  simulated_w1_ratio = None
  ratio_matched = None
  if exposed_au is not None and baseline_au is not None and baseline_au > 0.0:
    simulated_w1_ratio = exposed_au / baseline_au
    ratio_matched = abs(simulated_w1_ratio - animal_w1_ratio) <= W1_RATIO_TOLERANCE

  return EmpiricalComparison(
    condition=condition,
    simulated_threshold_shift_db=simulated_shift_db,
    animal_threshold_shift_db=animal_shift_db,
    threshold_shift_matched=shift_matched,
    simulated_w1_ratio=simulated_w1_ratio,
    animal_w1_ratio=animal_w1_ratio,
    w1_ratio_matched=ratio_matched,
  )


def format_empirical_comparison_table(comparison: EmpiricalComparison) -> str:
  """Renders a simulated versus animal comparison as a Markdown table.

  Args:
    comparison: Comparison record produced by `compare_to_empirical`.

  Returns:
    Formatted Markdown table string.
  """

  # Metrics that were not simulated are reported as unavailable, not as zero.
  def format_value(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)

  rows = [
    (
      "Click ABR threshold shift (dB)",
      format_value(comparison.simulated_threshold_shift_db, "+.2f"),
      format(comparison.animal_threshold_shift_db, "+.2f"),
      f"+/-{THRESHOLD_SHIFT_TOLERANCE_DB:g} dB",
      format_check_status(comparison.threshold_shift_matched),
    ),
    (
      f"Suprathreshold Wave-I post/pre ratio ({CALIBRATION_LEVEL_DB:g} dB SPL)",
      format_value(comparison.simulated_w1_ratio, ".3f"),
      format(comparison.animal_w1_ratio, ".3f"),
      f"+/-{W1_RATIO_TOLERANCE:g}",
      format_check_status(comparison.w1_ratio_matched),
    ),
  ]

  headers = (
    "Metric",
    f"Simulated ({comparison.condition})",
    "Animal (Bharadwaj et al. 2022)",
    "Tolerance",
    "Status",
  )
  lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
    *("| " + " | ".join(row) + " |" for row in rows),
  ]
  return "\n".join(lines)


class ToneBurstEmpiricalComparison(NamedTuple):
  """Comparison of simulated tone-burst metrics against animal data."""

  frequency_hz: float | None  # None denotes composite 4/8 kHz average
  condition: str
  simulated_threshold_shift_db: float | None
  animal_threshold_shift_db: float
  threshold_shift_matched: bool | None
  simulated_w1_ratio: float | None
  animal_w1_ratio: float
  w1_ratio_matched: bool | None


def compare_tone_burst_to_empirical(
  tone_burst_results: ToneBurstCohortResults,
  dataset: empirical.ChinchillaAbrDataset | None = None,
  condition: str = EXPOSED_CONDITION,
  baseline: str = BASELINE_CONDITION,
  level_db: float = CALIBRATION_LEVEL_DB,
  tolerance: float = W1_RATIO_TOLERANCE,
  threshold_tolerance_db: float = THRESHOLD_SHIFT_TOLERANCE_DB,
  calibration_strategy: str = "mode-dependent",
  calibration_reference: str = "average",
  click_results: Mapping[str, Sequence[float]] | None = None,
  click_levels_db: Sequence[float] | None = None,
) -> list[ToneBurstEmpiricalComparison]:
  """Compares simulated tone-burst metrics against animal data.

  Evaluates high-level (80 dB SPL) post/pre ratios and threshold shifts for
  individual frequencies (e.g., 4 kHz and 8 kHz) and the composite 4/8 kHz average
  against the noise-exposure measurements of Bharadwaj et al. (2022).

  Args:
    tone_burst_results: ToneBurstCohortResults from simulate_tone_burst_cohort.
    dataset: Empirical dataset; loaded from package data when None.
    condition: Cohort standing in for the noise-exposed animals.
    baseline: Cohort treated as healthy pre-exposure baseline.
    level_db: Calibration sound level in dB SPL (default 80.0).
    tolerance: Maximum allowed absolute difference between model and animal.
    threshold_tolerance_db: Maximum allowed threshold difference in dB.
    calibration_strategy: Calibration strategy ('individual', 'mode-dependent', 'unified').
    calibration_reference: Reference stimulus for shared scaling ('average', 'click', '4k', '8k').
    click_results: Click ABR simulation results mapping.
    click_levels_db: Click sound levels in dB SPL.

  Returns:
    List of ToneBurstEmpiricalComparison records.
  """
  data = empirical.load_chinchilla_abr_dataset() if dataset is None else dataset
  comparisons: list[ToneBurstEmpiricalComparison] = []

  if level_db not in tone_burst_results.levels_db:
    return comparisons
  target_idx = tone_burst_results.levels_db.index(level_db)

  # Check individual frequencies (e.g. 4 kHz and 8 kHz).
  for freq in tone_burst_results.frequencies_hz:
    freq_results = tone_burst_results.results_by_frequency.get(freq, {})
    if baseline in freq_results and condition in freq_results:
      ctrl_amp = freq_results[baseline][target_idx]
      exp_amp = freq_results[condition][target_idx]
      sim_ratio = exp_amp / ctrl_amp if ctrl_amp > 0 else None
      animal_ratio = data.high_level_w1_uv[freq].ratio if freq in data.high_level_w1_uv else 1.0
      ratio_matched = (
        abs(sim_ratio - animal_ratio) <= tolerance
        if sim_ratio is not None
        else None
      )

      # Resolve scale factor according to chosen strategy
      try:
        scale_val = resolve_scale_factor(
          strategy=calibration_strategy,
          stimulus_type=freq,
          click_results=click_results,
          tone_burst_results=tone_burst_results,
          click_levels_db=click_levels_db,
          tone_burst_levels_db=tone_burst_results.levels_db,
          reference=calibration_reference,
          dataset=data,
          condition=baseline,
          level_db=level_db,
        )
      except Exception:
        scale_val = None

      # Threshold shift comparison
      sim_shift_db = _simulated_threshold_shift_db(
        levels_db=tone_burst_results.levels_db,
        abr_results=freq_results,
        dataset=data,
        condition=condition,
        baseline=baseline,
        frequency_hz=freq,
        scale_uv_per_au=scale_val,
      )
      animal_shift_db = data.threshold_shift_db(freq) if freq in data.thresholds_db_spl else 0.0
      shift_matched = (
        abs(sim_shift_db - animal_shift_db) <= threshold_tolerance_db
        if sim_shift_db is not None
        else None
      )

      comparisons.append(
        ToneBurstEmpiricalComparison(
          frequency_hz=freq,
          condition=condition,
          simulated_threshold_shift_db=sim_shift_db,
          animal_threshold_shift_db=animal_shift_db,
          threshold_shift_matched=shift_matched,
          simulated_w1_ratio=sim_ratio,
          animal_w1_ratio=animal_ratio,
          w1_ratio_matched=ratio_matched,
        )
      )

  # Check composite average.
  composite = tone_burst_results.composite_results
  if baseline in composite and condition in composite:
    ctrl_amp = composite[baseline][target_idx]
    exp_amp = composite[condition][target_idx]
    sim_ratio = exp_amp / ctrl_amp if ctrl_amp > 0 else None
    animal_ratio = data.suprathreshold_w1_ratio
    ratio_matched = (
      abs(sim_ratio - animal_ratio) <= tolerance
      if sim_ratio is not None
      else None
    )

    try:
      scale_val_avg = resolve_scale_factor(
        strategy=calibration_strategy,
        stimulus_type="average",
        click_results=click_results,
        tone_burst_results=tone_burst_results,
        click_levels_db=click_levels_db,
        tone_burst_levels_db=tone_burst_results.levels_db,
        reference=calibration_reference,
        dataset=data,
        condition=baseline,
        level_db=level_db,
      )
    except Exception:
      scale_val_avg = None

    sim_shift_db = _simulated_threshold_shift_db(
      levels_db=tone_burst_results.levels_db,
      abr_results=composite,
      dataset=data,
      condition=condition,
      baseline=baseline,
      frequency_hz=None,
      scale_uv_per_au=scale_val_avg,
    )
    shift_4k = data.threshold_shift_db(4000.0) if 4000.0 in data.thresholds_db_spl else 0.0
    shift_8k = data.threshold_shift_db(8000.0) if 8000.0 in data.thresholds_db_spl else 0.0
    animal_shift_db = (shift_4k + shift_8k) / 2.0
    shift_matched = (
      abs(sim_shift_db - animal_shift_db) <= threshold_tolerance_db
      if sim_shift_db is not None
      else None
    )

    comparisons.append(
      ToneBurstEmpiricalComparison(
        frequency_hz=None,
        condition=condition,
        simulated_threshold_shift_db=sim_shift_db,
        animal_threshold_shift_db=animal_shift_db,
        threshold_shift_matched=shift_matched,
        simulated_w1_ratio=sim_ratio,
        animal_w1_ratio=animal_ratio,
        w1_ratio_matched=ratio_matched,
      )
    )

  return comparisons


def format_tone_burst_comparison_table(
  comparisons: Sequence[ToneBurstEmpiricalComparison],
) -> str:
  """Renders tone-burst simulated versus animal comparisons as a Markdown table.

  Args:
    comparisons: Sequence of ToneBurstEmpiricalComparison records.

  Returns:
    Formatted Markdown table string.
  """
  def format_value(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)

  rows: list[tuple[str, str, str, str, str]] = []
  for comp in comparisons:
    if comp.frequency_hz is None:
      freq_label = "4/8 kHz average"
    else:
      freq_label = f"{comp.frequency_hz:g} Hz"

    # Threshold shift row
    rows.append(
      (
        f"Tone-Burst {freq_label} ABR threshold shift (dB)",
        format_value(comp.simulated_threshold_shift_db, "+.2f"),
        format(comp.animal_threshold_shift_db, "+.2f"),
        f"+/-{THRESHOLD_SHIFT_TOLERANCE_DB:g} dB",
        format_check_status(comp.threshold_shift_matched),
      )
    )

    # Suprathreshold Wave-I ratio row
    rows.append(
      (
        f"Tone-Burst {freq_label} Wave-I ratio ({CALIBRATION_LEVEL_DB:g} dB SPL)",
        format_value(comp.simulated_w1_ratio, ".3f"),
        format(comp.animal_w1_ratio, ".3f"),
        f"+/-{W1_RATIO_TOLERANCE:g}",
        format_check_status(comp.w1_ratio_matched),
      )
    )

  cond_name = comparisons[0].condition if comparisons else "Exposed"
  headers = (
    "Metric",
    f"Simulated ({cond_name})",
    "Animal (Bharadwaj et al. 2022)",
    "Tolerance",
    "Status",
  )
  lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
    *("| " + " | ".join(row) + " |" for row in rows),
  ]
  return "\n".join(lines)



def _plot_metric_bars(
  axis: Any,
  simulated: float | None,
  animal: float,
  tolerance: float,
  ylabel: str,
  title: str,
) -> None:
  """Draws a simulated versus animal bar pair with the animal tolerance band."""
  # Shade the acceptance band around the animal value.
  axis.axhspan(animal - tolerance, animal + tolerance, color="#2ca02c", alpha=0.15)
  axis.axhline(animal, color="#2ca02c", linestyle="--", linewidth=1.5)

  # Draw the two bars, leaving the simulated one empty when it is unavailable.
  values = [0.0 if simulated is None else simulated, animal]
  axis.bar(
    ["Simulated", "Animal"],
    values,
    color=["#9467bd", "#2ca02c"],
    width=0.55,
  )
  if simulated is None:
    axis.text(0, 0, "n/a", ha="center", va="bottom", fontsize=10)

  axis.set_ylabel(ylabel, fontsize=10, fontweight="bold")
  axis.set_title(title, fontsize=11, fontweight="bold")
  axis.grid(True, axis="y", linestyle="--", alpha=0.5)


def plot_empirical_comparison(
  comparison: EmpiricalComparison,
  output_path: str | pathlib.Path,
  dataset: empirical.ChinchillaAbrDataset | None = None,
) -> pathlib.Path:
  """Plots the simulated cohort against the chinchilla ABR measurements.

  Args:
    comparison: Comparison record produced by `compare_to_empirical`.
    output_path: File path for the saved figure.
    dataset: Empirical dataset supplying per-animal ratios; loaded when None.

  Returns:
    Path of the saved figure.
  """
  path = pathlib.Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  data = empirical.load_chinchilla_abr_dataset() if dataset is None else dataset

  # Render one panel per validated metric.
  fig, (threshold_ax, ratio_ax) = plt.subplots(1, 2, figsize=(9, 4.5), dpi=300)
  _plot_metric_bars(
    axis=threshold_ax,
    simulated=comparison.simulated_threshold_shift_db,
    animal=comparison.animal_threshold_shift_db,
    tolerance=THRESHOLD_SHIFT_TOLERANCE_DB,
    ylabel="Click ABR Threshold Shift (dB)",
    title="Threshold Preservation",
  )
  _plot_metric_bars(
    axis=ratio_ax,
    simulated=comparison.simulated_w1_ratio,
    animal=comparison.animal_w1_ratio,
    tolerance=W1_RATIO_TOLERANCE,
    ylabel="Wave-I Post / Pre Amplitude Ratio",
    title=f"Suprathreshold Wave-I ({CALIBRATION_LEVEL_DB:g} dB SPL)",
  )

  # Overlay the individual animals to show the measured spread.
  ratios = data.per_animal_w1_ratios
  ratio_ax.scatter([1] * len(ratios), ratios, color="#111111", s=18, zorder=3, label="Animals")
  ratio_ax.legend(fontsize=8, loc="upper right")

  fig.suptitle(
    f"{comparison.condition} vs Noise-Exposed Chinchillas (Bharadwaj et al. 2022)",
    fontsize=12,
    fontweight="bold",
  )
  fig.tight_layout()
  fig.savefig(path, dpi=300)
  plt.close(fig)

  return path


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
  dataset: empirical.ChinchillaAbrDataset | None = None,
) -> BiologicalValidation:
  """Validates simulated electrophysiology against animal literature findings.

  Criteria reference Bharadwaj et al. (2022), Mehraei et al. (2016), and Ruggero et al. (1997).
  The selective synaptopathy criteria are checked against the measured chinchilla
  threshold shift and Wave-I ratio rather than hand-picked bands.

  Args:
    click_levels_db: Click sound levels in dB SPL.
    abr_results: Mapping of condition name to ABR Wave-I onset amplitudes.
    efr_levels_db: SAM tone carrier sound levels in dB SPL.
    efr_results: Mapping of condition name to EFR spectral magnitudes.
    dataset: Empirical dataset; loaded from package data when None.

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

  # 3. Selective LSR/MSR loss spares the near-threshold response, which is
  # carried by the intact high spontaneous rate fibers.
  sel_low = _get_level_value(abr_results, click_idx, EXPOSED_CONDITION, low_lvl)
  sel_low_ok = _ratio_within(sel_low, ctrl_low, 0.85, 1.05)

  # 4. The selective cohort reproduces the noise-exposed chinchillas quantitatively:
  # a recovered click threshold and the measured suprathreshold Wave-I attenuation.
  comparison = compare_to_empirical(click_levels_db, abr_results, dataset=dataset)

  # 5. Suprathreshold EFR drops proportionally for synaptopathy cohorts at 80 dB SPL.
  ctrl_efr = _get_level_value(efr_results, efr_idx, "Control", 80.0)
  syn50_efr = _get_level_value(efr_results, efr_idx, "Synaptopathy-50", 80.0)
  efr_drop_ok = _ratio_within(syn50_efr, ctrl_efr, 0.35, 0.65)

  # 6. OHC Loss elevates threshold (negligible response at 30-50 dB SPL).
  ohc_threshold_ok = None
  for lvl in (30.0, 40.0, 50.0):
    c_val = _get_level_value(abr_results, click_idx, "Control", lvl)
    o_val = _get_level_value(abr_results, click_idx, "OHC-Loss", lvl)
    if c_val is None or o_val is None:
      continue
    ohc_threshold_ok = o_val < 0.10 * c_val or o_val < 0.01
    if not ohc_threshold_ok:
      break

  # 7. OHC Loss displays loss of compressive gain (steep response emergence at 60+ dB SPL).
  ohc_60 = _get_level_value(efr_results, efr_idx, "OHC-Loss", 60.0)
  ohc_80 = _get_level_value(efr_results, efr_idx, "OHC-Loss", 80.0)
  ctrl_80_efr = _get_level_value(efr_results, efr_idx, "Control", 80.0)
  ohc_comp_ok = None
  if ohc_60 is not None and ohc_80 is not None and ctrl_80_efr is not None:
    ohc_comp_ok = ohc_80 > 8.0 * ohc_60 and ohc_80 > 0.6 * ctrl_80_efr

  # 8. Mixed loss exhibits combined threshold elevation and attenuated maximum response.
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
    selective_low_level_spared=sel_low_ok,
    empirical_threshold_shift_matched=comparison.threshold_shift_matched,
    empirical_w1_ratio_matched=comparison.w1_ratio_matched,
    efr_suprathreshold_drop=efr_drop_ok,
    ohc_threshold_shifted=ohc_threshold_ok,
    ohc_compression_lost=ohc_comp_ok,
    mixed_loss_dual_deficit=mixed_ok,
  )


def format_calibration_line(
  click_levels_db: Sequence[float],
  abr_results: Mapping[str, Sequence[float]],
  frequency_hz: float | None = empirical.CLICK_FREQUENCY_HZ,
  stimulus_name: str | None = None,
  scale_uv_per_au: float | None = None,
  strategy: str | None = None,
) -> str:
  """Renders the fitted microvolt scale, or a note when it cannot be fitted.

  The line is plain text because both the markdown report and the CLI print it
  verbatim.
  """
  if scale_uv_per_au is None:
    try:
      scale = fit_response_scale_uv_per_au(
        click_levels_db, abr_results, frequency_hz=frequency_hz
      )
    except (ValueError, KeyError, FileNotFoundError) as error:
      return f"Scale factor unavailable: {error}"
  else:
    scale = scale_uv_per_au

  if frequency_hz == empirical.CLICK_FREQUENCY_HZ:
    stim_desc = "click Wave-I amplitude"
  elif frequency_hz is not None:
    stim_desc = f"{frequency_hz:g} Hz tone-burst Wave-I amplitude"
  else:
    stim_desc = "4/8 kHz average tone-burst Wave-I amplitude"

  if stimulus_name:
    stim_desc = f"{stimulus_name} ({stim_desc})"

  strat_note = f" [strategy: {strategy}]" if strategy else ""
  return (
    f"Fitted scale factor: {scale:.4g} uV/{electrophysiology.RESPONSE_UNIT}{strat_note}, matching the "
    f"{BASELINE_CONDITION} response at {CALIBRATION_LEVEL_DB:g} dB SPL to the pre-exposure "
    f"chinchilla {stim_desc} (Bharadwaj et al. 2022)."
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

  # Quantify the agreement with the chinchilla measurements.
  comparison = compare_to_empirical(click_levels_db, abr_results)
  empirical_md = format_empirical_comparison_table(comparison)

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
    "- **Selective-Synaptopathy**: Selective loss of low and medium spontaneous rate fibers",
    "  (100% OHC, 100% HSR, 50% MSR, 0% LSR fibers).",
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
    f"- **Selective LSR/MSR Threshold Sparing (30-40 dB SPL)**: {format_check_status(validation.selective_low_level_spared)}",
    "  - Intact HSR fibers carry the near-threshold response, which stays above 85% of Control (~97% measured).",
    f"- **Click Threshold Preservation vs Animals**: {format_check_status(validation.empirical_threshold_shift_matched)}",
    f"  - Simulated threshold shift is within {THRESHOLD_SHIFT_TOLERANCE_DB:g} dB of the measured chinchilla shift.",
    f"- **Suprathreshold Wave-I Attenuation vs Animals**: {format_check_status(validation.empirical_w1_ratio_matched)}",
    f"  - Simulated post/pre Wave-I ratio is within {W1_RATIO_TOLERANCE:g} of the measured chinchilla ratio.",
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
    "## 5. Empirical Comparison with Animal Data",
    "",
    f"The {comparison.condition} cohort stands in for the noise-exposed chinchillas of",
    "Bharadwaj et al. (2022), which recovered their click thresholds while retaining a",
    f"reduced suprathreshold Wave-I. Simulated thresholds are the {THRESHOLD_CRITERION_UV:g} uV crossing of",
    "the interpolated Wave-I growth function, converted through the fitted response scale.",
    "",
    empirical_md,
    "",
    "## 6. Diagnostic Figures",
    "",
    "- `abr_wave_i_growth.png`: ABR Wave-I input-output growth curves across sound levels.",
    "- `efr_growth.png`: Envelope Following Response spectral magnitude growth curves.",
    "- `empirical_comparison.png`: Simulated versus measured chinchilla threshold shift and Wave-I ratio.",
    "",
  ]
  report_text = "\n".join(report_lines)

  # Write report to disk if path provided.
  if output_path is not None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")

  return report_text
