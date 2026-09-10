"""Electrophysiological response extraction for CARFAC simulations."""

from collections.abc import Sequence

import numpy as np

# Unit label for extracted responses. CARFAC neural activity patterns are
# dimensionless model output, not calibrated firing rates or recorded voltages,
# so responses are reported in arbitrary units and calibrated post hoc against
# animal recordings via `fit_microvolts_per_au`.
RESPONSE_UNIT: str = "AU"


def fit_microvolts_per_au(
  responses_au: Sequence[float],
  measured_uv: Sequence[float],
) -> float:
  """Fits the scale factor converting simulated responses into microvolts.

  Solves the least-squares proportional fit measured_uv ~ k * responses_au,
  which has no intercept because a zero model response corresponds to a zero
  recorded potential.

  Args:
    responses_au: Simulated responses in arbitrary units.
    measured_uv: Recorded amplitudes in microvolts, paired with responses_au.

  Returns:
    Scale factor in microvolts per arbitrary unit (> 0).
  """
  # Validate paired, non-empty, finite inputs.
  simulated = np.asarray(responses_au, dtype=np.float64)
  measured = np.asarray(measured_uv, dtype=np.float64)
  if simulated.ndim != 1 or measured.ndim != 1:
    raise ValueError(f"Inputs must be 1D, got shapes {simulated.shape} and {measured.shape}.")
  if simulated.size != measured.size:
    raise ValueError(f"Inputs must be equally long, got {simulated.size} and {measured.size}.")
  if simulated.size == 0:
    raise ValueError("Inputs cannot be empty.")
  if not (np.isfinite(simulated).all() and np.isfinite(measured).all()):
    raise ValueError("Inputs contain NaN or Inf values.")

  # Reject degenerate fits that would divide by a vanishing model response.
  denominator = float(np.sum(simulated * simulated))
  if denominator <= 0.0:
    raise ValueError("responses_au must contain a non-zero value.")

  scale = float(np.sum(simulated * measured) / denominator)
  if scale <= 0.0:
    raise ValueError(f"Fitted scale must be positive, got {scale}.")
  return scale


def compute_population_rate(naps: np.ndarray) -> np.ndarray:
  """Computes population auditory nerve activity across channels.

  Sums across cochlear channels: r(t) = sum_c naps(t, c). The result is in
  arbitrary units (`RESPONSE_UNIT`), not calibrated spike rates.

  Args:
    naps: Neural activity patterns array of shape (samples, n_channels) or (samples,).

  Returns:
    1D array of population response of shape (samples,), in arbitrary units.
  """
  # Reject non-array and string types.
  if isinstance(naps, str):
    raise TypeError(f"naps cannot be a string, got {type(naps).__name__}.")

  # Convert to float64 array.
  arr = np.asarray(naps, dtype=np.float64)

  # Validate finite numerical values.
  if not np.isfinite(arr).all():
    raise ValueError("naps contains NaN or Inf values.")

  # Handle 1D input array.
  if arr.ndim == 1:
    return arr.copy()

  # Validate 2D dimensions.
  if arr.ndim != 2:
    raise ValueError(f"naps must be 1D or 2D, got shape {arr.shape}.")

  # Sum across channels along axis 1.
  return np.sum(arr, axis=1)


def extract_wave_i_amplitude(
  population_rate: np.ndarray,
  sample_rate: int,
  stimulus_onset_s: float = 0.005,
  window_s: float = 0.008,
  mode: str = "baseline_to_peak",
) -> float:
  """Extracts Compound Action Potential (CAP) / ABR Wave-I onset amplitude.

  Finds the onset response after stimulus_onset_s within window_s. Computes
  either baseline-subtracted peak amplitude or peak-to-trough amplitude.

  Args:
    population_rate: 1D array of population response in arbitrary units.
    sample_rate: Sampling rate in Hz.
    stimulus_onset_s: Stimulus onset time in seconds.
    window_s: Analysis window duration following onset in seconds.
    mode: Extraction mode ('baseline_to_peak' or 'peak_to_trough').

  Returns:
    Extracted Wave-I amplitude in arbitrary units (>= 0.0).
  """
  # Reject non-array and string types.
  if isinstance(population_rate, str):
    raise TypeError(f"population_rate cannot be a string, got {type(population_rate).__name__}.")

  # Validate numerical parameters.
  if sample_rate <= 0:
    raise ValueError("sample_rate must be positive.")
  if stimulus_onset_s < 0.0:
    raise ValueError("stimulus_onset_s must be non-negative.")
  if window_s <= 0.0:
    raise ValueError("window_s must be positive.")
  if mode not in ("baseline_to_peak", "peak_to_trough"):
    raise ValueError(f"Unsupported mode '{mode}'. Must be 'baseline_to_peak' or 'peak_to_trough'.")

  # Convert input rate to 1D float64 array.
  rate = np.asarray(population_rate, dtype=np.float64)
  if rate.ndim == 2 and rate.shape[1] == 1:
    rate = rate[:, 0]
  if rate.ndim != 1:
    raise ValueError(f"population_rate must be 1D, got shape {rate.shape}.")

  # Validate finite numerical values.
  if not np.isfinite(rate).all():
    raise ValueError("population_rate contains NaN or Inf values.")

  # Handle empty signal.
  if rate.size == 0:
    return 0.0

  # Compute onset and window sample indices.
  onset_idx = int(round(stimulus_onset_s * sample_rate))
  window_samples = int(round(window_s * sample_rate))
  end_idx = onset_idx + window_samples

  # Signal ended before or at stimulus onset.
  if onset_idx >= rate.size:
    return 0.0

  # Extract onset window.
  window = rate[onset_idx : min(rate.size, end_idx)]
  if window.size == 0:
    return 0.0

  # Compute peak in window.
  peak = float(np.max(window))

  # Compute amplitude according to selected mode.
  if mode == "baseline_to_peak":
    baseline = float(np.mean(rate[:onset_idx])) if onset_idx > 0 else float(np.min(window))
    return float(max(0.0, peak - baseline))

  # mode == "peak_to_trough": find minimum at or after the peak within window.
  peak_idx = int(np.argmax(window))
  trough = float(np.min(window[peak_idx:]))
  return float(max(0.0, peak - trough))


def extract_efr_amplitude(
  population_rate: np.ndarray,
  sample_rate: int,
  fm_hz: float,
  steady_state_start_s: float = 0.05,
  single_sided: bool = False,
) -> float:
  """Extracts Envelope Following Response (EFR) spectral magnitude.

  Computes the Fourier magnitude at modulation frequency fm_hz of the
  steady-state population response, normalized by the window length.

  Args:
    population_rate: 1D array of population response in arbitrary units.
    sample_rate: Sampling rate in Hz.
    fm_hz: Modulation frequency in Hz.
    steady_state_start_s: Start time of steady-state window in seconds.
    single_sided: If True, returns single-sided spectral amplitude (2 * |X| / N).
      If False, returns normalized DFT magnitude (|X| / N).

  Returns:
    Extracted EFR amplitude at modulation frequency fm_hz, in arbitrary units (>= 0.0).
  """
  # Reject non-array and string types.
  if isinstance(population_rate, str):
    raise TypeError(f"population_rate cannot be a string, got {type(population_rate).__name__}.")

  # Validate parameters.
  if sample_rate <= 0:
    raise ValueError("sample_rate must be positive.")
  if fm_hz <= 0.0:
    raise ValueError("fm_hz must be positive.")
  if fm_hz >= sample_rate / 2.0:
    raise ValueError("fm_hz must be below Nyquist frequency.")
  if steady_state_start_s < 0.0:
    raise ValueError("steady_state_start_s must be non-negative.")

  # Convert input rate to 1D float64 array.
  rate = np.asarray(population_rate, dtype=np.float64)
  if rate.ndim == 2 and rate.shape[1] == 1:
    rate = rate[:, 0]
  if rate.ndim != 1:
    raise ValueError(f"population_rate must be 1D, got shape {rate.shape}.")

  # Validate finite numerical values.
  if not np.isfinite(rate).all():
    raise ValueError("population_rate contains NaN or Inf values.")

  # Slicing steady state response.
  start_idx = int(round(steady_state_start_s * sample_rate))
  if start_idx >= rate.size:
    return 0.0

  steady_signal = rate[start_idx:]
  n_samples = steady_signal.size
  if n_samples == 0:
    return 0.0

  # Remove DC baseline to prevent spectral leakage at non-integer cycle boundaries.
  signal_demeaned = steady_signal - np.mean(steady_signal)

  # Compute Discrete-Time Fourier Transform at modulation frequency fm_hz.
  t = np.arange(n_samples, dtype=np.float64) / sample_rate
  dft_val = np.sum(signal_demeaned * np.exp(-2j * np.pi * fm_hz * t))
  raw_magnitude = float(np.abs(dft_val) / n_samples)

  # Scale to single-sided amplitude if requested.
  if single_sided:
    return float(2.0 * raw_magnitude)
  return raw_magnitude
