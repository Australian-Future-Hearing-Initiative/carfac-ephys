"""Constants and calibration utilities for CARFAC electrophysiology."""

from typing import overload

import numpy as np

# Default acoustic sampling rate in Hz.
DEFAULT_SAMPLE_RATE: int = 32000

# Full scale reference level: 0 dB FS = 104 dB SPL.
DYNAMIC_RANGE_DB: float = 104.0


# The dB conversions below return a scalar for a scalar and an array for an
# array. The overloads tell type checkers which one a call site gets.
@overload
def db_spl_to_amplitude(db_spl: float) -> float: ...
@overload
def db_spl_to_amplitude(db_spl: np.ndarray) -> np.ndarray: ...
def db_spl_to_amplitude(
  db_spl: float | np.ndarray,
) -> float | np.ndarray:
  """Converts sound pressure level in dB SPL to digital linear amplitude.

  Uses reference 0 dB FS = 104 dB SPL:
  A = 10 ** ((db_spl - 104.0) / 20.0).

  Args:
    db_spl: Sound pressure level in dB SPL.

  Returns:
    Digital amplitude (scalar or array).
  """
  # Compute linear digital amplitude.
  return 10.0 ** ((db_spl - DYNAMIC_RANGE_DB) / 20.0)


@overload
def amplitude_to_db_spl(amplitude: float | np.floating) -> float: ...
@overload
def amplitude_to_db_spl(amplitude: np.ndarray) -> np.ndarray: ...
def amplitude_to_db_spl(
  amplitude: float | np.floating | np.ndarray,
) -> float | np.ndarray:
  """Converts digital linear amplitude to sound pressure level in dB SPL.

  Args:
    amplitude: Digital linear amplitude.

  Returns:
    Sound pressure level in dB SPL.
  """
  # Safe log computation handling non-positive amplitudes.
  amp_arr = np.asarray(amplitude)
  with np.errstate(divide="ignore", invalid="ignore"):
    db = 20.0 * np.log10(np.where(amp_arr > 0.0, amp_arr, np.nan)) + DYNAMIC_RANGE_DB

  # Return scalar or array matching input type.
  db_clean = np.nan_to_num(db, nan=-np.inf)
  if np.ndim(amplitude) == 0:
    return float(db_clean)
  return db_clean
