"""Outer/middle-ear pre-cochlear filtering for CARFAC stimuli.

CARFAC models the cochlea and auditory nerve but has no middle-ear stage, so a
waveform handed to it is treated as though it were already at the stapes.
Applying a middle-ear transfer function first makes the cochlear input the
stapes signal it is meant to be.

The filters here are the Verhulst et al. band-pass approximations of the human
middle-ear transfer function, fitted to Puria's human cadaver measurements.
They are human fits: applying one while comparing against the chinchilla
dataset mixes species, which is a modelling choice to make deliberately rather
than by default.
"""

import dataclasses

import numpy as np
from scipy import signal

from carfac_ephys import constants

# Gain of the Puria human cadaver fit, in dB, shared by every variant below.
DEFAULT_PURIA_GAIN_DB: float = 9.0


@dataclasses.dataclass(frozen=True)
class MiddleEarFilter:
  """A band-pass approximation of the human middle-ear transfer function.

  Attributes:
    name: The published variant this reproduces.
    low_cut_hz: Lower -3 dB corner frequency.
    high_cut_hz: Upper -3 dB corner frequency.
    gain_factor: Fixed linear gain the published variant applies on top of the
      Puria gain.
    puria_gain_db: Gain of the Puria human cadaver fit, in dB.
  """

  name: str
  low_cut_hz: float
  high_cut_hz: float
  gain_factor: float
  puria_gain_db: float = DEFAULT_PURIA_GAIN_DB

  @property
  def linear_gain(self) -> float:
    """Total linear gain folded into the filter's numerator."""
    return self.gain_factor * 10.0 ** (self.puria_gain_db / 20.0)


VERHULST_2012 = MiddleEarFilter("Verhulst_2012", 100.0, 3000.0, 2.0)
VERHULST_2015 = MiddleEarFilter("Verhulst_2015", 600.0, 3000.0, 2.0)
VERHULST_2018 = MiddleEarFilter("Verhulst_2018", 600.0, 4000.0, 1.0)

# Name to filter, for the CLI and anything else selecting one by string.
FILTERS: dict[str, MiddleEarFilter] = {
  variant.name: variant for variant in (VERHULST_2012, VERHULST_2015, VERHULST_2018)
}

DEFAULT_FILTER: MiddleEarFilter = VERHULST_2018


def design(
  middle_ear_filter: MiddleEarFilter = DEFAULT_FILTER,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
  """Designs the filter, folding its gain into the numerator coefficients.

  Args:
    middle_ear_filter: Variant to design.
    sample_rate: Sampling rate in Hz.

  Returns:
    The (b, a) coefficients, with `linear_gain` already applied to b.

  Raises:
    ValueError: If the corner frequencies are not ordered below Nyquist.
  """
  if sample_rate <= 0:
    raise ValueError("sample_rate must be positive.")
  nyquist_hz = sample_rate / 2.0
  if not 0.0 < middle_ear_filter.low_cut_hz < middle_ear_filter.high_cut_hz < nyquist_hz:
    raise ValueError(
      f"{middle_ear_filter.name} needs 0 < {middle_ear_filter.low_cut_hz} < "
      f"{middle_ear_filter.high_cut_hz} < Nyquist ({nyquist_hz}) Hz; "
      f"sample_rate={sample_rate} is too low."
    )

  # scipy ships no type information, so `butter` is inferred as returning either
  # (b, a) or the three-element zpk form. Indexing is valid for both, and pins
  # the dtype while we are here.
  coefficients = signal.butter(
    1,
    Wn=[middle_ear_filter.low_cut_hz, middle_ear_filter.high_cut_hz],
    btype="bandpass",
    output="ba",
    fs=sample_rate,
  )
  b_coeffs = np.asarray(coefficients[0], dtype=np.float64)
  a_coeffs = np.asarray(coefficients[1], dtype=np.float64)

  return b_coeffs * middle_ear_filter.linear_gain, a_coeffs


def apply_middle_ear(
  waveform: np.ndarray,
  middle_ear_filter: MiddleEarFilter = DEFAULT_FILTER,
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
  """Filters a stimulus waveform through the middle ear.

  Args:
    waveform: 1-D stimulus in the same units as `stimuli.generate_click`.
    middle_ear_filter: Variant to apply.
    sample_rate: Sampling rate in Hz.

  Returns:
    The filtered waveform, same shape and dtype convention as the input.

  Raises:
    ValueError: If the waveform is not 1-D or is not finite.
  """
  wave = np.asarray(waveform, dtype=np.float64)
  if wave.ndim != 1:
    raise ValueError(f"waveform must be 1-D, got shape {wave.shape}.")
  if not np.isfinite(wave).all():
    raise ValueError("waveform contains NaN or Inf values.")
  if wave.size == 0:
    return wave

  b_coeffs, a_coeffs = design(middle_ear_filter, sample_rate)
  return np.asarray(signal.lfilter(b_coeffs, a_coeffs, wave), dtype=np.float64)
