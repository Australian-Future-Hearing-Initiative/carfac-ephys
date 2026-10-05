"""Tests for unitary response convolution."""

import numpy as np
import pytest

from carfac_ephys.constants import DEFAULT_SAMPLE_RATE
from carfac_ephys.unitary_response import UnitaryResponse, convolve, derive, synthetic


def _impulse(n_samples: int, index: int) -> np.ndarray:
  rate = np.zeros(n_samples)
  rate[index] = 1.0
  return rate


class TestUnitaryResponseKernel:
  @pytest.mark.parametrize(
    ("kernel", "message"),
    [
      (np.zeros((4, 2)), "1-D"),
      (np.zeros(0), "empty"),
      (np.full(8, np.nan), "NaN"),
    ],
    ids=["two-dimensional", "empty", "not-finite"],
  )
  def test_malformed_kernels_are_rejected(self, kernel, message):
    with pytest.raises(ValueError, match=message):
      UnitaryResponse(kernel=kernel, sample_rate=DEFAULT_SAMPLE_RATE, source="test")

  def test_duration_follows_the_sample_rate(self):
    response = synthetic(sample_rate=DEFAULT_SAMPLE_RATE, duration_s=0.01)
    assert response.duration_s == pytest.approx(0.01, abs=1.0 / DEFAULT_SAMPLE_RATE)

  def test_resampling_preserves_duration(self):
    original = synthetic(sample_rate=100_000, duration_s=0.01)
    resampled = original.resampled_to(DEFAULT_SAMPLE_RATE)

    assert resampled.sample_rate == DEFAULT_SAMPLE_RATE
    assert resampled.duration_s == pytest.approx(original.duration_s, rel=0.02)
    assert "resampled" in resampled.source

  def test_resampling_to_the_same_rate_is_a_no_op(self):
    original = synthetic()
    assert original.resampled_to(original.sample_rate) is original


class TestConvolve:
  def test_an_impulse_reproduces_the_kernel(self):
    # Convolving a unit impulse with the UR must return the UR itself,
    # positioned at the impulse. This is the whole contract.
    response = synthetic()
    rate = _impulse(2048, 100)
    evoked = convolve(rate, response, DEFAULT_SAMPLE_RATE)

    assert evoked.shape == rate.shape
    kernel = response.kernel
    np.testing.assert_allclose(evoked[100 : 100 + kernel.size], kernel, atol=1e-12)
    assert np.allclose(evoked[:100], 0.0)

  def test_latency_is_preserved_not_centred(self):
    # mode='same' would shift the peak earlier by half the kernel; the UR
    # carries the wave I-to-V delays, so that shift must not happen.
    response = synthetic()
    onset = 500
    evoked = convolve(_impulse(4096, onset), response, DEFAULT_SAMPLE_RATE)

    assert int(np.argmax(np.abs(evoked))) >= onset

  def test_convolution_is_additive_across_channels(self):
    # The paper relies on this: convolving each CF then summing equals summing
    # then convolving once, which is why we convolve the population rate.
    response = synthetic()
    rng = np.random.default_rng(0)
    channels = rng.normal(size=(1024, 6))

    summed_first = convolve(channels.sum(axis=1), response, DEFAULT_SAMPLE_RATE)
    convolved_first = sum(convolve(channels[:, c], response, DEFAULT_SAMPLE_RATE) for c in range(6))

    np.testing.assert_allclose(summed_first, convolved_first, atol=1e-9)

  def test_output_length_matches_input(self):
    response = synthetic()
    assert convolve(np.zeros(777), response, DEFAULT_SAMPLE_RATE).shape == (777,)

  @pytest.mark.parametrize(
    "bad", [np.zeros((8, 2)), np.full(8, np.inf)], ids=["two-dimensional", "not-finite"]
  )
  def test_malformed_rates_are_rejected(self, bad):
    with pytest.raises(ValueError):
      convolve(bad, synthetic(), DEFAULT_SAMPLE_RATE)


class TestDerive:
  def test_round_trip_recovers_a_known_kernel(self):
    # Make a potential from a kernel we know, then recover the kernel from it.
    rng = np.random.default_rng(1)
    rate = np.abs(rng.normal(size=4096))
    truth = synthetic(duration_s=0.004)
    potential = convolve(rate, truth, DEFAULT_SAMPLE_RATE)

    recovered = derive(
      potential,
      rate,
      DEFAULT_SAMPLE_RATE,
      source="round-trip test",
      duration_s=0.004,
      regularisation=1e-6,
    )

    correlation = np.corrcoef(recovered.kernel, truth.kernel)[0, 1]
    assert correlation > 0.99

  def test_mismatched_lengths_are_rejected(self):
    with pytest.raises(ValueError, match="same length"):
      derive(np.zeros(100), np.zeros(99), DEFAULT_SAMPLE_RATE, source="test")

  def test_silent_rate_is_rejected(self):
    with pytest.raises(ValueError, match="no energy"):
      derive(np.zeros(256), np.zeros(256), DEFAULT_SAMPLE_RATE, source="test")

  def test_provenance_survives(self):
    rng = np.random.default_rng(2)
    rate = np.abs(rng.normal(size=1024))
    potential = convolve(rate, synthetic(duration_s=0.004), DEFAULT_SAMPLE_RATE)
    derived = derive(potential, rate, DEFAULT_SAMPLE_RATE, source="Dau et al. 2000 Fig 2")

    assert derived.source == "Dau et al. 2000 Fig 2"
