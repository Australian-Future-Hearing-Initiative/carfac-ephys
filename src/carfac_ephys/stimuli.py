"""Calibrated acoustic stimulus generation for CARFAC electrophysiology."""

import numpy as np

from carfac_ephys import constants


def generate_click(
  duration_s: float = 0.03,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  peak_db_spl: float = 80.0,
  pulse_width_s: float = 0.0001,
  delay_s: float = 0.005,
) -> np.ndarray:
  """Generates a rectangular click pulse calibrated to a specified peak dB SPL.

  Args:
    duration_s: Total duration in seconds.
    sample_rate: Sampling rate in Hz.
    peak_db_spl: Peak sound pressure level in dB SPL.
    pulse_width_s: Rectangular pulse duration in seconds.
    delay_s: Onset delay in seconds.

  Returns:
    1D array of stimulus waveform values.
  """
  # Validate input parameters.
  if duration_s <= 0.0:
    raise ValueError("duration_s must be positive.")
  if sample_rate <= 0:
    raise ValueError("sample_rate must be positive.")
  if pulse_width_s <= 0.0:
    raise ValueError("pulse_width_s must be positive.")
  if delay_s < 0.0:
    raise ValueError("delay_s must be non-negative.")
  if delay_s + pulse_width_s > duration_s:
    raise ValueError("delay_s + pulse_width_s exceeds duration_s.")

  # Compute indices and allocate buffer.
  n_samples = int(round(duration_s * sample_rate))
  waveform = np.zeros(n_samples, dtype=np.float64)
  start_idx = int(round(delay_s * sample_rate))
  pulse_samples = max(1, int(round(pulse_width_s * sample_rate)))
  end_idx = min(n_samples, start_idx + pulse_samples)

  # Calibrate peak amplitude against CARFAC reference.
  amplitude = float(constants.db_spl_to_amplitude(peak_db_spl))
  waveform[start_idx:end_idx] = amplitude

  return waveform


def _compute_raised_cosine_ramp(
  n_samples: int,
  n_ramp: int,
) -> np.ndarray:
  """Computes a symmetric raised cosine onset and offset ramp."""
  # Bound ramp length to half of buffer.
  n_ramp = min(n_samples // 2, max(0, n_ramp))

  # Return uniform envelope if ramp duration is zero.
  if n_ramp <= 0:
    return np.ones(n_samples, dtype=np.float64)

  # Form raised cosine onset from 0 to 1 and reverse for offset.
  ramp = np.ones(n_samples, dtype=np.float64)
  ramp_phases = np.linspace(0.0, np.pi, n_ramp)
  ramp_onset = 0.5 * (1.0 - np.cos(ramp_phases))
  ramp_offset = ramp_onset[::-1]

  # Apply ramps to buffer boundaries.
  ramp[:n_ramp] = ramp_onset
  ramp[-n_ramp:] = ramp_offset

  return ramp


def generate_sam_tone(
  fc_hz: float = 2000.0,
  fm_hz: float = 100.0,
  depth: float = 1.0,
  duration_s: float = 0.2,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  target_db_spl: float = 70.0,
  ramp_s: float = 0.01,
) -> np.ndarray:
  """Generates a sinusoidally amplitude-modulated (SAM) tone.

  Generates s(t) = A * (1 + depth * sin(2*pi*fm*t - pi/2)) * sin(2*pi*fc*t)
  with raised cosine ramped onset/offset, calibrated so carrier RMS matches
  target_db_spl.

  Args:
    fc_hz: Carrier frequency in Hz.
    fm_hz: Modulation frequency in Hz.
    depth: Modulation depth (>= 0.0).
    duration_s: Total duration in seconds.
    sample_rate: Sampling rate in Hz.
    target_db_spl: Target carrier RMS level in dB SPL.
    ramp_s: Duration of raised cosine onset and offset ramps in seconds.

  Returns:
    1D array of modulated stimulus waveform values.
  """
  # Validate input parameters.
  if duration_s <= 0.0:
    raise ValueError("duration_s must be positive.")
  if sample_rate <= 0:
    raise ValueError("sample_rate must be positive.")
  if fc_hz <= 0.0 or fm_hz <= 0.0:
    raise ValueError("Frequencies fc_hz and fm_hz must be positive.")
  if fc_hz >= sample_rate / 2.0:
    raise ValueError("fc_hz must be below Nyquist frequency.")
  if depth < 0.0:
    raise ValueError("depth must be non-negative.")
  if ramp_s < 0.0:
    raise ValueError("ramp_s must be non-negative.")
  if 2.0 * ramp_s > duration_s:
    raise ValueError("2 * ramp_s cannot exceed duration_s.")

  # Compute time array and carrier peak amplitude.
  n_samples = int(round(duration_s * sample_rate))
  t = np.arange(n_samples, dtype=np.float64) / sample_rate
  carrier_rms = float(constants.db_spl_to_amplitude(target_db_spl))
  carrier_peak = carrier_rms * np.sqrt(2.0)

  # Synthesize modulated waveform.
  envelope = 1.0 + depth * np.sin(2.0 * np.pi * fm_hz * t - np.pi / 2.0)
  carrier = np.sin(2.0 * np.pi * fc_hz * t)
  signal = carrier_peak * envelope * carrier

  # Apply onset and offset tapering.
  n_ramp = int(round(ramp_s * sample_rate))
  ramp = _compute_raised_cosine_ramp(n_samples, n_ramp)
  signal *= ramp

  return signal


def _compute_linear_ramp(
  n_samples: int,
  n_ramp: int,
) -> np.ndarray:
  """Computes a symmetric linear onset and offset ramp."""
  # Bound ramp length to half of buffer.
  n_ramp = min(n_samples // 2, max(0, n_ramp))

  # Return uniform envelope if ramp duration is zero.
  if n_ramp <= 0:
    return np.ones(n_samples, dtype=np.float64)

  # Form linear onset from 0 to 1 and reverse for offset.
  ramp = np.ones(n_samples, dtype=np.float64)
  ramp_onset = np.linspace(0.0, 1.0, n_ramp, dtype=np.float64)
  ramp_offset = ramp_onset[::-1]

  # Apply ramps to buffer boundaries.
  ramp[:n_ramp] = ramp_onset
  ramp[-n_ramp:] = ramp_offset

  return ramp


def generate_tone_burst(
  frequency_hz: float = 4000.0,
  duration_s: float = 0.005,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  peak_db_spl: float = 80.0,
  ramp_s: float = 0.0005,
  delay_s: float = 0.0,
  polarity: int = 1,
  phase: float = 0.0,
) -> np.ndarray:
  """Generates a calibrated tone burst with linear onset and offset ramps.

  Level is calibrated relative to wave peak (0 dB FS = 104 dB SPL) using
  constants.db_spl_to_amplitude.

  Args:
    frequency_hz: Carrier frequency in Hz (default 4000.0).
    duration_s: Burst duration in seconds (default 0.005 = 5 ms).
    sample_rate: Sampling rate in Hz.
    peak_db_spl: Peak sound pressure level in dB SPL (default 80.0).
    ramp_s: Duration of linear onset and offset ramps in seconds (default 0.0005 = 0.5 ms).
    delay_s: Onset delay prior to burst in seconds.
    polarity: Stimulus polarity (+1 or -1).
    phase: Initial sine carrier phase in radians.

  Returns:
    1D array of stimulus waveform values.
  """
  # Validate input parameters.
  if duration_s <= 0.0:
    raise ValueError("duration_s must be positive.")
  if sample_rate <= 0:
    raise ValueError("sample_rate must be positive.")
  if frequency_hz <= 0.0:
    raise ValueError("frequency_hz must be positive.")
  if frequency_hz >= sample_rate / 2.0:
    raise ValueError("frequency_hz must be below Nyquist frequency.")
  if ramp_s < 0.0:
    raise ValueError("ramp_s must be non-negative.")
  if 2.0 * ramp_s > duration_s:
    raise ValueError("2 * ramp_s cannot exceed duration_s.")
  if delay_s < 0.0:
    raise ValueError("delay_s must be non-negative.")
  if polarity not in (1, -1):
    raise ValueError("polarity must be +1 or -1.")

  # Compute indices and allocate buffer.
  burst_samples = int(round(duration_s * sample_rate))
  delay_samples = int(round(delay_s * sample_rate))
  total_samples = delay_samples + burst_samples
  waveform = np.zeros(total_samples, dtype=np.float64)

  # Synthesize linear ramped carrier.
  t = np.arange(burst_samples, dtype=np.float64) / sample_rate
  carrier = np.sin(2.0 * np.pi * frequency_hz * t + phase)
  n_ramp = int(round(ramp_s * sample_rate))
  envelope = _compute_linear_ramp(burst_samples, n_ramp)

  # Calibrate peak amplitude against CARFAC reference.
  amplitude = float(constants.db_spl_to_amplitude(peak_db_spl))
  burst = float(polarity) * amplitude * envelope * carrier
  waveform[delay_samples:] = burst

  return waveform


def generate_tone_burst_train(
  frequency_hz: float = 4000.0,
  duration_s: float = 0.005,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  peak_db_spl: float = 80.0,
  ramp_s: float = 0.0005,
  rate_hz: float = 20.0,
  n_repetitions: int = 500,
  delay_s: float = 0.0,
  alternating_polarity: bool = True,
  phase: float = 0.0,
) -> np.ndarray:
  """Generates a continuous sequence of tone bursts at a fixed repetition rate.

  Repeats tone bursts at `rate_hz` with optional alternating polarity (+1, -1)
  across successive repetitions.

  Args:
    frequency_hz: Carrier frequency in Hz (default 4000.0).
    duration_s: Duration of each tone burst in seconds (default 0.005).
    sample_rate: Sampling rate in Hz.
    peak_db_spl: Peak sound pressure level in dB SPL (default 80.0).
    ramp_s: Duration of linear onset and offset ramps in seconds (default 0.0005).
    rate_hz: Repetition rate in Hz (default 20.0).
    n_repetitions: Number of burst repetitions (default 500).
    delay_s: Onset delay within each repetition interval in seconds.
    alternating_polarity: Whether to alternate polarity between repetitions.
    phase: Initial sine carrier phase in radians.

  Returns:
    1D array of concatenated stimulus waveform values.
  """
  # Validate input parameters.
  if rate_hz <= 0.0:
    raise ValueError("rate_hz must be positive.")
  if n_repetitions <= 0:
    raise ValueError("n_repetitions must be positive.")

  period_s = 1.0 / rate_hz
  if delay_s + duration_s > period_s:
    raise ValueError("delay_s + duration_s cannot exceed repetition period (1 / rate_hz).")

  period_samples = int(round(sample_rate / rate_hz))

  # Generate positive burst prototype placed at delay_s.
  pos_burst = generate_tone_burst(
    frequency_hz=frequency_hz,
    duration_s=duration_s,
    sample_rate=sample_rate,
    peak_db_spl=peak_db_spl,
    ramp_s=ramp_s,
    delay_s=delay_s,
    polarity=1,
    phase=phase,
  )

  # Template for a single repetition period.
  template_pos = np.zeros(period_samples, dtype=np.float64)
  template_pos[: len(pos_burst)] = pos_burst

  # Assemble full repetition sequence.
  if alternating_polarity:
    template_neg = -template_pos
    reps = np.empty((n_repetitions, period_samples), dtype=np.float64)
    reps[0::2] = template_pos
    reps[1::2] = template_neg
    return reps.reshape(-1)

  return np.tile(template_pos, n_repetitions)
