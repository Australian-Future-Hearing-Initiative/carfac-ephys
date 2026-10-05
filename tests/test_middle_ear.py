"""Tests for the Verhulst middle-ear pre-cochlear filter."""

import numpy as np
import pytest
from scipy import signal

from carfac_ephys import stimuli
from carfac_ephys.constants import DEFAULT_SAMPLE_RATE
from carfac_ephys.middle_ear import (
  FILTERS,
  VERHULST_2018,
  MiddleEarFilter,
  apply_middle_ear,
  design,
)


class TestFilterDesign:
  """The designed response must match the published variant."""

  @pytest.mark.parametrize("name", sorted(FILTERS), ids=sorted(FILTERS))
  def test_peak_gain_matches_the_published_gain(self, name):
    # Peak gain is gain_factor plus the Puria gain, by construction.
    variant = FILTERS[name]
    b_coeffs, a_coeffs = design(variant, DEFAULT_SAMPLE_RATE)
    _, response = signal.freqz(b_coeffs, a_coeffs, worN=8192, fs=DEFAULT_SAMPLE_RATE)
    peak_db = 20.0 * np.log10(np.abs(response).max())
    expected_db = 20.0 * np.log10(variant.gain_factor) + variant.puria_gain_db

    assert peak_db == pytest.approx(expected_db, abs=0.05)

  @pytest.mark.parametrize("name", sorted(FILTERS), ids=sorted(FILTERS))
  def test_corner_frequencies_are_three_db_down(self, name):
    variant = FILTERS[name]
    b_coeffs, a_coeffs = design(variant, DEFAULT_SAMPLE_RATE)
    _, corners = signal.freqz(
      b_coeffs,
      a_coeffs,
      worN=[variant.low_cut_hz, variant.high_cut_hz],
      fs=DEFAULT_SAMPLE_RATE,
    )
    _, full = signal.freqz(b_coeffs, a_coeffs, worN=8192, fs=DEFAULT_SAMPLE_RATE)
    peak = np.abs(full).max()

    for corner in np.abs(corners):
      assert corner == pytest.approx(peak / np.sqrt(2.0), rel=0.02)

  def test_dc_is_rejected(self):
    # A band-pass must pass nothing at DC, or a click gains a constant offset.
    b_coeffs, a_coeffs = design(VERHULST_2018, DEFAULT_SAMPLE_RATE)
    _, response = signal.freqz(b_coeffs, a_coeffs, worN=[0.0], fs=DEFAULT_SAMPLE_RATE)

    assert np.abs(response[0]) == pytest.approx(0.0, abs=1e-12)

  def test_corners_above_nyquist_are_rejected(self):
    with pytest.raises(ValueError, match="too low"):
      design(VERHULST_2018, sample_rate=4000)

  def test_unordered_corners_are_rejected(self):
    inverted = MiddleEarFilter("inverted", 4000.0, 600.0, 1.0)
    with pytest.raises(ValueError, match="Nyquist"):
      design(inverted, DEFAULT_SAMPLE_RATE)


class TestApplyMiddleEar:
  """Applying the filter to a stimulus."""

  def test_shape_is_preserved(self):
    click = stimuli.generate_click()
    filtered = apply_middle_ear(click, VERHULST_2018, DEFAULT_SAMPLE_RATE)

    assert filtered.shape == click.shape
    assert np.isfinite(filtered).all()

  def test_a_click_is_changed(self):
    # The point of the filter is that it changes the stimulus.
    click = stimuli.generate_click()
    filtered = apply_middle_ear(click, VERHULST_2018, DEFAULT_SAMPLE_RATE)

    assert not np.allclose(filtered, click)

  def test_low_frequency_energy_is_attenuated(self):
    # A 100 Hz tone is well below the 600 Hz corner; a 1 kHz tone is in band.
    duration_s, fs = 0.1, DEFAULT_SAMPLE_RATE
    t = np.arange(int(duration_s * fs)) / fs
    low = np.sin(2.0 * np.pi * 100.0 * t)
    mid = np.sin(2.0 * np.pi * 1000.0 * t)

    low_gain = np.abs(apply_middle_ear(low)).max() / np.abs(low).max()
    mid_gain = np.abs(apply_middle_ear(mid)).max() / np.abs(mid).max()

    assert low_gain < mid_gain

  def test_silence_stays_silent(self):
    assert np.allclose(apply_middle_ear(np.zeros(256)), 0.0)

  def test_empty_waveform(self):
    assert apply_middle_ear(np.zeros(0)).shape == (0,)

  @pytest.mark.parametrize(
    "bad", [np.zeros((4, 2)), np.full(8, np.nan)], ids=["two-dimensional", "not-finite"]
  )
  def test_malformed_waveforms_are_rejected(self, bad):
    with pytest.raises(ValueError):
      apply_middle_ear(bad)
