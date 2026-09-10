"""CARFAC model wrapper for biophysical hearing impairment electrophysiology."""

from collections.abc import Sequence
import copy
from typing import NamedTuple

from carfac.jax import carfac
import jax.numpy as jnp
import numpy as np

from carfac_ephys import constants


class FiberRetention(NamedTuple):
  """Auditory nerve fiber retention fractions for HSR, MSR, and LSR fibers."""

  hsr: float = 1.0
  msr: float = 1.0
  lsr: float = 1.0


def _broadcast_ohc_health(
  ohc_health: float | Sequence[float] | np.ndarray,
  n_channels: int,
) -> np.ndarray:
  """Validates and broadcasts OHC health values to a channel array."""
  # Reject strings.
  if isinstance(ohc_health, str):
    raise TypeError(f"ohc_health cannot be a string, got {type(ohc_health).__name__}.")

  # Convert scalar health to broadcasted channel array.
  if isinstance(ohc_health, (int, float, np.number)) and not isinstance(
    ohc_health, (bool, np.bool_)
  ):
    val = float(ohc_health)
    if not 0.0 <= val <= 1.0:
      raise ValueError(f"ohc_health must be in [0, 1], got {val}.")
    return np.full(n_channels, val, dtype=np.float32)

  # Validate array shape and range.
  if isinstance(ohc_health, (Sequence, np.ndarray)):
    try:
      health = np.asarray(ohc_health, dtype=np.float32)
    except (ValueError, TypeError) as e:
      raise TypeError("Failed to convert ohc_health elements to float.") from e
    if health.shape != (n_channels,):
      raise ValueError(f"Expected ohc_health shape ({n_channels},), got {health.shape}.")
    if not np.isfinite(health).all() or np.any(health < 0.0) or np.any(health > 1.0):
      raise ValueError("All ohc_health values must be in [0, 1].")
    return health

  raise TypeError(f"Unsupported ohc_health type: {type(ohc_health).__name__}.")


def _normalize_fiber_retention(
  fiber_retention: float | FiberRetention | Sequence[float] | np.ndarray,
) -> np.ndarray:
  """Validates and converts fiber retention factors to a length-3 float array."""
  # Reject strings.
  if isinstance(fiber_retention, str):
    raise TypeError(f"fiber_retention cannot be a string, got {type(fiber_retention).__name__}.")

  # Handle scalar retention.
  if isinstance(fiber_retention, (int, float, np.number)) and not isinstance(
    fiber_retention, (bool, np.bool_)
  ):
    val = float(fiber_retention)
    factors = np.full(3, val, dtype=np.float32)

  # Handle sequence or 1D array (including FiberRetention named tuple).
  elif isinstance(fiber_retention, (Sequence, np.ndarray)):
    try:
      ret_arr = np.asarray(fiber_retention, dtype=np.float32)
    except (ValueError, TypeError) as e:
      raise TypeError("Failed to convert fiber_retention elements to float.") from e
    if ret_arr.shape != (3,):
      raise ValueError(f"fiber_retention sequence must have length 3, got shape {ret_arr.shape}.")
    factors = ret_arr

  else:
    raise TypeError(f"Unsupported fiber_retention type: {type(fiber_retention).__name__}.")

  # Validate value bounds in [0, 1].
  if not np.isfinite(factors).all() or np.any(factors < 0.0) or np.any(factors > 1.0):
    raise ValueError(f"All fiber_retention factors must be in [0, 1], got {factors}.")

  return factors


class CarfacModel:
  """CARFAC model wrapper for simulating cochlear electrophysiology."""

  def __init__(
    self,
    params: carfac.CarfacDesignParameters,
    hypers: carfac.CarfacHypers,
    weights: carfac.CarfacWeights,
    state: carfac.CarfacState,
    fs: int,
  ) -> None:
    # Store CARFAC configuration and state objects.
    self.params = params
    self.hypers = hypers
    self.weights = weights
    self.state = state
    self.fs = fs
    self._initial_state = copy.deepcopy(state)

  @property
  def n_channels(self) -> int:
    """Returns the number of cochlear frequency channels."""
    return self.hypers.ears[0].n_ch

  @property
  def pole_freqs(self) -> np.ndarray:
    """Returns characteristic frequency poles in Hz."""
    return np.asarray(self.hypers.ears[0].pole_freqs)

  @property
  def ohc_health(self) -> np.ndarray:
    """Returns outer hair cell health array."""
    return np.asarray(self.weights.ears[0].car.ohc_health)

  @property
  def n_fibers(self) -> np.ndarray:
    """Returns auditory nerve fiber counts array of shape (n_channels, 3)."""
    return np.asarray(self.weights.ears[0].syn.n_fibers)

  def reset(self) -> None:
    """Resets the model state to initial resting conditions."""
    self.state = copy.deepcopy(self._initial_state)

  def run(self, waveform: np.ndarray) -> np.ndarray:
    """Runs a stimulus waveform and returns neural activity patterns (NAPs).

    Args:
      waveform: 1D or 2D single-channel stimulus array.

    Returns:
      NAPs array of shape (samples, n_channels).
    """
    # Convert input to float32 array.
    wave_arr = np.asarray(waveform, dtype=np.float32)

    # Validate input dimensions.
    if not (wave_arr.ndim == 1 or (wave_arr.ndim == 2 and wave_arr.shape[1] == 1)):
      raise ValueError(f"waveform must be 1D or 2D with 1 channel, got shape {wave_arr.shape}.")

    # Check for NaN or Inf values.
    if not np.isfinite(wave_arr).all():
      raise ValueError("waveform contains NaN or Inf values.")

    # Handle zero-sample input.
    if wave_arr.shape[0] == 0:
      return np.zeros((0, self.n_channels), dtype=np.float32)

    # Run CARFAC segment.
    output = carfac.run_segment(
      wave_arr,
      self.hypers,
      self.weights,
      self.state,
    )

    # Update state and extract naps for primary ear.
    self.state = output.state
    return np.asarray(output.naps[:, :, 0], dtype=np.float32)


def build_model(
  ohc_health: float | Sequence[float] | np.ndarray = 1.0,
  fiber_retention: float | FiberRetention | Sequence[float] | np.ndarray = 1.0,
  fs: int = constants.DEFAULT_SAMPLE_RATE,
) -> CarfacModel:
  """Builds and initializes a CarfacModel with specified biophysical health.

  Args:
    ohc_health: Outer hair cell health in [0, 1], scalar or array matching channel count.
    fiber_retention: Auditory nerve fiber retention in [0, 1] across HSR, MSR, LSR.
    fs: Sampling rate in Hz.

  Returns:
    Configured CarfacModel ready for simulation.
  """
  # Validate sampling rate.
  if fs <= 0:
    raise ValueError(f"fs must be positive, got {fs}.")

  # Initialize design parameters with two-capacitor IHC and delay buffer.
  params = carfac.CarfacDesignParameters(fs=fs)
  for ear in params.ears:
    ear.ihc.ihc_style = "two_cap_with_syn"
    ear.car.use_delay_buffer = True

  # Design and initialize CARFAC structures.
  hypers, weights, state = carfac.design_and_init_carfac(params)
  n_ch = hypers.ears[0].n_ch

  # Broadcast and validate OHC health across channels.
  ohc_arr = _broadcast_ohc_health(ohc_health, n_ch)

  # Normalize and validate fiber retention factors for [HSR, MSR, LSR].
  fiber_factors = _normalize_fiber_retention(fiber_retention)

  # Apply OHC health and fiber retention to ear weights.
  for ear_weights in weights.ears:
    ear_weights.car.ohc_health = jnp.asarray(ohc_arr, dtype=jnp.float32)
    ear_weights.syn.n_fibers = ear_weights.syn.n_fibers * jnp.asarray(
      fiber_factors, dtype=jnp.float32
    )

  return CarfacModel(
    params=params,
    hypers=hypers,
    weights=weights,
    state=state,
    fs=fs,
  )
