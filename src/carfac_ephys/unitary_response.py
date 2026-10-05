"""Unitary response convolution for simulating far-field evoked potentials.

Implements the framework of Temboury-Gutierrez, Encina-Llamas and Dau (2024),
JASA 155, 1799-1812, after Dau (2003): a far-field auditory evoked potential is
the compound auditory-nerve activity convolved with a single unitary response
(UR) function that is independent of level and characteristic frequency,

    AEP(t) = compound_AN(t) * UR(t)

The UR is not derived from theory. It is obtained by deconvolving a measured,
averaged ABR with the simulated compound AN response to the same stimulus, so
it absorbs the propagation to the electrode and the contributions of the later
brainstem nuclei that generate waves III and V. See `derive` in this module.

A UR is only valid for the AN front end it was derived against: the published
work used Bruce et al. (2018) at 100 kHz with 200 CFs and 32 000 fibres, which
is not what CARFAC produces here. Treat any UR as data belonging to a specific
model, and record where it came from.
"""

import dataclasses
import pathlib

import numpy as np
from scipy import signal

from carfac_ephys import constants

# The published UR spans roughly 10 ms, long enough to carry waves I to V.
DEFAULT_DURATION_S: float = 0.010


@dataclasses.dataclass(frozen=True)
class UnitaryResponse:
  """A unitary response kernel and the provenance needed to interpret it.

  Attributes:
    kernel: The UR waveform, one sample per `sample_rate` tick.
    sample_rate: Sampling rate the kernel is defined at, in Hz.
    source: Where this kernel came from. Carry it: a UR is only meaningful
      for the AN front end and stimulus it was derived against.
    units: Units of the convolved output, given an input in arbitrary units.
  """

  kernel: np.ndarray
  sample_rate: int
  source: str
  units: str = "arbitrary"

  def __post_init__(self) -> None:
    """Rejects kernels that cannot describe a unitary response."""
    if self.kernel.ndim != 1:
      raise ValueError(f"kernel must be 1-D, got shape {self.kernel.shape}.")
    if self.kernel.size == 0:
      raise ValueError("kernel must not be empty.")
    if not np.isfinite(self.kernel).all():
      raise ValueError("kernel contains NaN or Inf values.")
    if self.sample_rate <= 0:
      raise ValueError("sample_rate must be positive.")

  @property
  def duration_s(self) -> float:
    """Kernel length in seconds."""
    return self.kernel.size / self.sample_rate

  def resampled_to(self, sample_rate: int) -> "UnitaryResponse":
    """Returns this UR resampled to another rate, preserving its integral.

    The published UR is defined at the 100 kHz rate of the Bruce et al. AN
    model; this project runs CARFAC at a different rate, so a UR almost always
    needs resampling before use.
    """
    if sample_rate <= 0:
      raise ValueError("sample_rate must be positive.")
    if sample_rate == self.sample_rate:
      return self
    resampled = signal.resample_poly(self.kernel, sample_rate, self.sample_rate)
    return dataclasses.replace(
      self,
      kernel=np.asarray(resampled, dtype=np.float64),
      sample_rate=sample_rate,
      source=f"{self.source} (resampled {self.sample_rate} -> {sample_rate} Hz)",
    )


def synthetic(
  sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
  duration_s: float = DEFAULT_DURATION_S,
) -> UnitaryResponse:
  """Builds a PLACEHOLDER kernel for exercising the plumbing.

  This is NOT the unitary response of Temboury-Gutierrez et al. (2024), which
  is not published numerically. It is a damped oscillation with roughly the
  right duration and a few extrema, so that the convolution path can be tested
  and plotted before a real UR is available. Any result produced with it is
  about the code, not about the auditory system.
  """
  time_s = np.arange(int(round(duration_s * sample_rate))) / sample_rate
  # ~1 kHz oscillation decaying over the kernel, giving several extrema in 10 ms.
  kernel = np.exp(-time_s / 0.0025) * np.sin(2.0 * np.pi * 1000.0 * time_s)
  return UnitaryResponse(
    kernel=kernel / np.abs(kernel).max(),
    sample_rate=sample_rate,
    source="SYNTHETIC PLACEHOLDER - not a derived unitary response",
  )


def load(path: pathlib.Path | str, sample_rate: int, source: str) -> UnitaryResponse:
  """Loads a UR kernel from a one-column text or .npy file."""
  file_path = pathlib.Path(path)
  if file_path.suffix == ".npy":
    kernel = np.load(file_path)
  else:
    kernel = np.loadtxt(file_path)
  return UnitaryResponse(
    kernel=np.asarray(kernel, dtype=np.float64).ravel(),
    sample_rate=sample_rate,
    source=source,
  )


def convolve(
  population_rate: np.ndarray,
  unitary_response: UnitaryResponse,
  sample_rate: int,
) -> np.ndarray:
  """Convolves compound AN activity with a UR to give an evoked potential.

  Args:
    population_rate: 1-D compound AN activity, as `compute_population_rate`
      returns it.
    unitary_response: The kernel to convolve with. Resampled to `sample_rate`
      if it was defined at another rate.
    sample_rate: Sampling rate of `population_rate` in Hz.

  Returns:
    The evoked potential, same length as the input, on the same time axis.

  Raises:
    ValueError: If the input is not 1-D or not finite.
  """
  rate = np.asarray(population_rate, dtype=np.float64)
  if rate.ndim != 1:
    raise ValueError(f"population_rate must be 1-D, got shape {rate.shape}.")
  if not np.isfinite(rate).all():
    raise ValueError("population_rate contains NaN or Inf values.")

  kernel = unitary_response.resampled_to(sample_rate).kernel
  # 'full' truncated to the input length, NOT mode='same': 'same' centres the
  # kernel and shifts every wave earlier by half its length, which would
  # destroy exactly the latencies the UR exists to carry.
  return np.convolve(rate, kernel)[: rate.size]
