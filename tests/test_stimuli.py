"""Tests for constants and calibrated stimulus generation."""

import numpy as np
import pytest

from carfac_ephys.constants import (
  DEFAULT_SAMPLE_RATE,
  DYNAMIC_RANGE_DB,
  amplitude_to_db_spl,
  db_spl_to_amplitude,
)
from carfac_ephys.stimuli import generate_click, generate_sam_tone


class TestConstants:
  """Tests for acoustic constants and calibration conversions."""

  def test_constants_values(self):
    assert DEFAULT_SAMPLE_RATE == 32000
    assert DYNAMIC_RANGE_DB == 104.0

  def test_db_spl_to_amplitude_reference(self):
    # 104 dB SPL must map exactly to digital full scale 1.0.
    assert np.isclose(db_spl_to_amplitude(104.0), 1.0)
    # 84 dB SPL is -20 dB FS -> amplitude 0.1.
    assert np.isclose(db_spl_to_amplitude(84.0), 0.1)
    # 64 dB SPL is -40 dB FS -> amplitude 0.01.
    assert np.isclose(db_spl_to_amplitude(64.0), 0.01)

  def test_amplitude_to_db_spl_roundtrip(self):
    # Round-trip conversions must be identity.
    for level in [30.0, 50.0, 70.0, 80.0, 104.0]:
      amp = db_spl_to_amplitude(level)
      recovered = amplitude_to_db_spl(amp)
      assert np.isclose(recovered, level)

  def test_array_conversion(self):
    levels = np.array([64.0, 84.0, 104.0])
    amps = db_spl_to_amplitude(levels)
    assert np.allclose(amps, [0.01, 0.1, 1.0])
    recovered = amplitude_to_db_spl(amps)
    assert np.allclose(recovered, levels)


class TestGenerateClick:
  """Tests for rectangular click stimulus generation."""

  def test_default_properties(self):
    click = generate_click()
    # Default duration: 0.03 s at 32000 Hz = 960 samples.
    assert click.shape == (960,)
    assert click.dtype == np.float64

  def test_timing_and_width(self):
    fs = 32000
    delay_s = 0.005
    pulse_width_s = 0.0001
    click = generate_click(
      duration_s=0.02,
      sample_rate=fs,
      peak_db_spl=80.0,
      pulse_width_s=pulse_width_s,
      delay_s=delay_s,
    )

    # Delay is 0.005 * 32000 = 160 samples.
    start_idx = int(round(delay_s * fs))
    expected_width = max(1, int(round(pulse_width_s * fs)))
    end_idx = start_idx + expected_width

    # Samples prior to pulse must be strictly zero.
    assert np.all(click[:start_idx] == 0.0)
    # Samples during pulse must be non-zero and positive.
    assert np.all(click[start_idx:end_idx] > 0.0)
    # Samples after pulse must be strictly zero.
    assert np.all(click[end_idx:] == 0.0)
    # Exactly pulse width samples must be non-zero.
    assert np.count_nonzero(click) == expected_width

  def test_peak_amplitude_calibration(self):
    # Click peak amplitude is calibrated from peak_db_spl.
    for level in [30.0, 50.0, 70.0, 80.0, 104.0]:
      click = generate_click(peak_db_spl=level)
      expected_peak = db_spl_to_amplitude(level)
      assert np.isclose(np.max(click), expected_peak)

  def test_invalid_parameters(self):
    # Duration must be positive.
    with pytest.raises(ValueError):
      generate_click(duration_s=0.0)
    with pytest.raises(ValueError):
      generate_click(duration_s=-0.01)

    # Sample rate must be positive.
    with pytest.raises(ValueError):
      generate_click(sample_rate=0)

    # Pulse width must be positive.
    with pytest.raises(ValueError):
      generate_click(pulse_width_s=0.0)

    # Delay cannot be negative.
    with pytest.raises(ValueError):
      generate_click(delay_s=-0.001)

    # Pulse must fit within duration.
    with pytest.raises(ValueError):
      generate_click(duration_s=0.01, delay_s=0.009, pulse_width_s=0.005)


class TestGenerateSamTone:
  """Tests for sinusoidally amplitude-modulated tone stimulus generation."""

  def test_default_properties(self):
    sam = generate_sam_tone()
    # Default duration: 0.2 s at 32000 Hz = 6400 samples.
    assert sam.shape == (6400,)
    assert sam.dtype == np.float64

  def test_carrier_rms_calibration(self):
    # With depth=0 (pure carrier) and no ramp, RMS must match target_db_spl.
    fs = 32000
    for level in [40.0, 60.0, 70.0, 80.0]:
      tone = generate_sam_tone(
        fc_hz=2000.0,
        fm_hz=100.0,
        depth=0.0,
        duration_s=0.2,
        sample_rate=fs,
        target_db_spl=level,
        ramp_s=0.0,
      )
      rms = np.sqrt(np.mean(tone**2))
      expected_rms = db_spl_to_amplitude(level)
      assert np.isclose(rms, expected_rms, rtol=1e-3)
      assert np.isclose(amplitude_to_db_spl(rms), level, atol=0.05)

  def test_modulation_envelope_scaling(self):
    # Carrier peak amplitude A = sqrt(2) * db_spl_to_amplitude(target_db_spl).
    # For 100% modulation (depth=1.0), peak amplitude in steady state is 2 * A.
    level = 70.0
    carrier_rms = db_spl_to_amplitude(level)
    carrier_peak = carrier_rms * np.sqrt(2.0)
    expected_sam_peak = 2.0 * carrier_peak

    # Tone with minimal ramp.
    sam = generate_sam_tone(
      fc_hz=2000.0,
      fm_hz=100.0,
      depth=1.0,
      duration_s=0.2,
      target_db_spl=level,
      ramp_s=0.01,
    )
    # Peak amplitude in steady state closely approaches 2 * A (within 0.5% due
    # to discrete sampling phase offset between carrier and modulator).
    steady_state = sam[int(0.02 * 32000) : int(0.18 * 32000)]
    assert np.isclose(np.max(steady_state), expected_sam_peak, rtol=5e-3)

  def test_raised_cosine_ramp(self):
    fs = 32000
    ramp_s = 0.01
    duration_s = 0.1
    sam = generate_sam_tone(
      fc_hz=2000.0,
      fm_hz=100.0,
      depth=1.0,
      duration_s=duration_s,
      sample_rate=fs,
      target_db_spl=70.0,
      ramp_s=ramp_s,
    )

    # First and last samples must be zero due to cosine ramp.
    assert np.isclose(sam[0], 0.0, atol=1e-7)
    assert np.isclose(sam[-1], 0.0, atol=1e-7)

    # Ramp-free version.
    sam_unramped = generate_sam_tone(
      fc_hz=2000.0,
      fm_hz=100.0,
      depth=1.0,
      duration_s=duration_s,
      sample_rate=fs,
      target_db_spl=70.0,
      ramp_s=0.0,
    )
    # Steady state region must match unramped tone.
    n_ramp = int(round(ramp_s * fs))
    assert np.allclose(sam[n_ramp:-n_ramp], sam_unramped[n_ramp:-n_ramp])

  def test_invalid_parameters(self):
    # Duration must be positive.
    with pytest.raises(ValueError):
      generate_sam_tone(duration_s=0.0)

    # Carrier frequency above Nyquist.
    with pytest.raises(ValueError):
      generate_sam_tone(fc_hz=17000.0, sample_rate=32000)

    # Modulation depth negative.
    with pytest.raises(ValueError):
      generate_sam_tone(depth=-0.5)

    # Ramp exceeds half duration.
    with pytest.raises(ValueError):
      generate_sam_tone(duration_s=0.02, ramp_s=0.015)

    # Non-positive sample rate.
    with pytest.raises(ValueError):
      generate_sam_tone(sample_rate=0)

    # Non-positive frequencies.
    with pytest.raises(ValueError):
      generate_sam_tone(fc_hz=0.0)
    with pytest.raises(ValueError):
      generate_sam_tone(fm_hz=-10.0)

    # Negative ramp.
    with pytest.raises(ValueError):
      generate_sam_tone(ramp_s=-0.01)

  def test_exact_formula_match(self):
    fc = 1500.0
    fm = 50.0
    depth = 0.8
    duration = 0.05
    fs = 32000
    level = 65.0
    ramp = 0.005

    signal = generate_sam_tone(
      fc_hz=fc,
      fm_hz=fm,
      depth=depth,
      duration_s=duration,
      sample_rate=fs,
      target_db_spl=level,
      ramp_s=ramp,
    )

    # Manual analytical calculation.
    n_samples = int(round(duration * fs))
    t = np.arange(n_samples) / fs
    carrier_rms = db_spl_to_amplitude(level)
    carrier_peak = carrier_rms * np.sqrt(2.0)
    expected_env = 1.0 + depth * np.sin(2.0 * np.pi * fm * t - np.pi / 2.0)
    expected_carrier = np.sin(2.0 * np.pi * fc * t)
    expected = carrier_peak * expected_env * expected_carrier

    n_ramp = int(round(ramp * fs))
    ramp_phases = np.linspace(0.0, np.pi, n_ramp)
    ramp_on = 0.5 * (1.0 - np.cos(ramp_phases))
    ramp_win = np.ones(n_samples)
    ramp_win[:n_ramp] = ramp_on
    ramp_win[-n_ramp:] = ramp_on[::-1]
    expected *= ramp_win

    assert np.allclose(signal, expected, atol=1e-12)

  def test_alternative_sampling_rates(self):
    for fs in [16000, 44100, 48000]:
      click = generate_click(sample_rate=fs, duration_s=0.01, delay_s=0.002)
      assert click.shape == (int(round(0.01 * fs)),)

      sam = generate_sam_tone(sample_rate=fs, duration_s=0.05, ramp_s=0.005)
      assert sam.shape == (int(round(0.05 * fs)),)


class TestEdgeCasesAndExports:
  """Tests for boundary conditions, non-positive amplitudes, and package exports."""

  def test_zero_delay_click(self):
    click = generate_click(duration_s=0.01, delay_s=0.0, pulse_width_s=0.0001)
    # Starts immediately at index 0.
    assert click[0] > 0.0

  def test_tiny_pulse_width_enforces_one_sample(self):
    click = generate_click(duration_s=0.01, delay_s=0.002, pulse_width_s=1e-9)
    # Enforces max(1, ...) sample width.
    assert np.count_nonzero(click) == 1

  def test_non_positive_amplitude_db_conversion(self):
    assert amplitude_to_db_spl(0.0) == -np.inf
    assert amplitude_to_db_spl(-1.0) == -np.inf
    arr = np.array([0.0, -0.5, 1.0])
    res = amplitude_to_db_spl(arr)
    assert res[0] == -np.inf
    assert res[1] == -np.inf
    assert np.isclose(res[2], 104.0)

  def test_package_exports(self):
    import carfac_ephys

    assert hasattr(carfac_ephys, "DEFAULT_SAMPLE_RATE")
    assert hasattr(carfac_ephys, "DYNAMIC_RANGE_DB")
    assert hasattr(carfac_ephys, "generate_click")
    assert hasattr(carfac_ephys, "generate_sam_tone")
    assert hasattr(carfac_ephys, "db_spl_to_amplitude")
    assert hasattr(carfac_ephys, "amplitude_to_db_spl")
