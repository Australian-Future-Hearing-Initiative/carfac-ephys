"""Tests for electrophysiological response extraction (Wave-I and EFR)."""

import numpy as np
import pytest

from carfac_ephys.carfac_model import build_model
from carfac_ephys.electrophysiology import (
  compute_population_rate,
  extract_efr_amplitude,
  extract_wave_i_amplitude,
  fit_microvolts_per_au,
)
from carfac_ephys.stimuli import generate_click, generate_sam_tone


class TestComputePopulationRate:
  """Tests for compute_population_rate."""

  def test_summation_across_channels(self):
    # 100 samples across 4 channels with known channel weights.
    n_samples, n_channels = 100, 4
    naps = np.ones((n_samples, n_channels), dtype=np.float32)
    pop_rate = compute_population_rate(naps)
    assert pop_rate.shape == (n_samples,)
    assert np.allclose(pop_rate, 4.0)

  def test_heterogeneous_channel_rates(self):
    naps = np.array(
      [
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [0.0, 0.0, 0.0],
      ],
      dtype=np.float64,
    )
    pop_rate = compute_population_rate(naps)
    assert np.allclose(pop_rate, [6.0, 4.5, 0.0])

  def test_1d_input_handling(self):
    rate_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    pop_rate = compute_population_rate(rate_1d)
    assert np.allclose(pop_rate, rate_1d)

  def test_empty_input(self):
    empty_2d = np.zeros((0, 10), dtype=np.float32)
    pop_rate = compute_population_rate(empty_2d)
    assert pop_rate.shape == (0,)

  def test_invalid_dimensions(self):
    with pytest.raises(ValueError, match="1D or 2D"):
      compute_population_rate(np.zeros((5, 5, 5)))

  def test_invalid_types_and_values(self):
    with pytest.raises(TypeError):
      compute_population_rate("invalid")
    with pytest.raises(ValueError, match="NaN or Inf"):
      compute_population_rate(np.array([[1.0, np.nan], [2.0, 3.0]]))


class TestExtractWaveIAmplitude:
  """Tests for extract_wave_i_amplitude."""

  def test_synthetic_pulse_baseline_to_peak(self):
    fs = 32000
    n_samples = int(0.02 * fs)
    pop_rate = np.full(n_samples, 2.0, dtype=np.float64)

    # Insert onset pulse at t = 0.007 s (after stimulus onset 0.005 s).
    pulse_idx = int(0.007 * fs)
    pop_rate[pulse_idx] = 12.0

    amp = extract_wave_i_amplitude(
      pop_rate,
      sample_rate=fs,
      stimulus_onset_s=0.005,
      window_s=0.008,
      mode="baseline_to_peak",
    )
    assert np.isclose(amp, 10.0)

  def test_synthetic_pulse_peak_to_trough(self):
    fs = 32000
    n_samples = int(0.02 * fs)
    pop_rate = np.full(n_samples, 5.0, dtype=np.float64)

    # Insert peak and subsequent trough.
    peak_idx = int(0.007 * fs)
    trough_idx = int(0.009 * fs)
    pop_rate[peak_idx] = 15.0
    pop_rate[trough_idx] = 3.0

    amp = extract_wave_i_amplitude(
      pop_rate,
      sample_rate=fs,
      stimulus_onset_s=0.005,
      window_s=0.008,
      mode="peak_to_trough",
    )
    assert np.isclose(amp, 12.0)

  def test_silent_and_zero_response(self):
    fs = 32000
    silent = np.zeros(int(0.02 * fs), dtype=np.float64)
    amp = extract_wave_i_amplitude(silent, sample_rate=fs)
    assert amp == 0.0

    empty = np.array([], dtype=np.float64)
    assert extract_wave_i_amplitude(empty, sample_rate=fs) == 0.0

  def test_signal_shorter_than_onset(self):
    fs = 32000
    short_signal = np.ones(int(0.002 * fs), dtype=np.float64)
    amp = extract_wave_i_amplitude(short_signal, sample_rate=fs, stimulus_onset_s=0.005)
    assert amp == 0.0

  def test_zero_onset_time(self):
    fs = 32000
    pop_rate = np.array([1.0, 5.0, 2.0, 1.0], dtype=np.float64)
    amp = extract_wave_i_amplitude(pop_rate, sample_rate=fs, stimulus_onset_s=0.0, window_s=0.001)
    assert amp > 0.0

  def test_invalid_parameters(self):
    valid_rate = np.ones(100)
    with pytest.raises(ValueError, match="sample_rate"):
      extract_wave_i_amplitude(valid_rate, sample_rate=0)
    with pytest.raises(ValueError, match="stimulus_onset_s"):
      extract_wave_i_amplitude(valid_rate, sample_rate=32000, stimulus_onset_s=-0.001)
    with pytest.raises(ValueError, match="window_s"):
      extract_wave_i_amplitude(valid_rate, sample_rate=32000, window_s=0.0)
    with pytest.raises(ValueError, match="mode"):
      extract_wave_i_amplitude(valid_rate, sample_rate=32000, mode="invalid")
    with pytest.raises(ValueError, match="NaN or Inf"):
      extract_wave_i_amplitude(np.array([1.0, np.nan]), sample_rate=32000)


class TestExtractEFRAmplitude:
  """Tests for extract_efr_amplitude."""

  def test_pure_sinusoid_magnitude(self):
    fs = 32000
    duration_s = 0.2
    fm_hz = 100.0
    mod_amp = 4.0
    baseline = 10.0

    t = np.arange(int(duration_s * fs)) / fs
    pop_rate = baseline + mod_amp * np.cos(2.0 * np.pi * fm_hz * t)

    # Raw DFT magnitude normalized by window length should equal mod_amp / 2.
    efr_raw = extract_efr_amplitude(
      pop_rate,
      sample_rate=fs,
      fm_hz=fm_hz,
      steady_state_start_s=0.05,
      single_sided=False,
    )
    assert np.isclose(efr_raw, mod_amp / 2.0, rtol=1e-3)

    # Single-sided magnitude should equal mod_amp.
    efr_single = extract_efr_amplitude(
      pop_rate,
      sample_rate=fs,
      fm_hz=fm_hz,
      steady_state_start_s=0.05,
      single_sided=True,
    )
    assert np.isclose(efr_single, mod_amp, rtol=1e-3)

  def test_frequency_selectivity(self):
    fs = 32000
    duration_s = 0.2
    fm_target = 100.0
    t = np.arange(int(duration_s * fs)) / fs
    pop_rate = 5.0 + 3.0 * np.sin(2.0 * np.pi * fm_target * t)

    # Use 0.10s window (integer cycles for 50 Hz, 100 Hz, and 200 Hz).
    efr_target = extract_efr_amplitude(
      pop_rate, sample_rate=fs, fm_hz=fm_target, steady_state_start_s=0.10
    )
    efr_off1 = extract_efr_amplitude(
      pop_rate, sample_rate=fs, fm_hz=50.0, steady_state_start_s=0.10
    )
    efr_off2 = extract_efr_amplitude(
      pop_rate, sample_rate=fs, fm_hz=200.0, steady_state_start_s=0.10
    )

    assert efr_target > 1.0
    assert np.isclose(efr_off1, 0.0, atol=1e-3)
    assert np.isclose(efr_off2, 0.0, atol=1e-3)

  def test_linearity_with_modulation_amplitude(self):
    fs = 32000
    t = np.arange(int(0.2 * fs)) / fs
    rate_1 = 5.0 + 2.0 * np.cos(2.0 * np.pi * 100.0 * t)
    rate_2 = 5.0 + 4.0 * np.cos(2.0 * np.pi * 100.0 * t)

    efr_1 = extract_efr_amplitude(rate_1, sample_rate=fs, fm_hz=100.0)
    efr_2 = extract_efr_amplitude(rate_2, sample_rate=fs, fm_hz=100.0)

    assert np.isclose(efr_2, 2.0 * efr_1, rtol=1e-3)

  def test_constant_dc_has_zero_efr(self):
    fs = 32000
    constant_rate = np.full(int(0.2 * fs), 15.0, dtype=np.float64)
    efr = extract_efr_amplitude(constant_rate, sample_rate=fs, fm_hz=100.0)
    assert np.isclose(efr, 0.0, atol=1e-6)

  def test_silent_and_empty_inputs(self):
    fs = 32000
    silent = np.zeros(int(0.2 * fs), dtype=np.float64)
    assert extract_efr_amplitude(silent, sample_rate=fs, fm_hz=100.0) == 0.0
    assert extract_efr_amplitude(np.array([]), sample_rate=fs, fm_hz=100.0) == 0.0

  def test_steady_state_exceeds_duration(self):
    fs = 32000
    short_rate = np.ones(int(0.02 * fs))
    assert (
      extract_efr_amplitude(short_rate, sample_rate=fs, fm_hz=100.0, steady_state_start_s=0.05)
      == 0.0
    )

  def test_stop_time_excludes_trailing_samples(self):
    fs = 32000
    fm_hz = 100.0
    t = np.arange(int(0.2 * fs)) / fs
    pop_rate = 5.0 + 2.0 * np.cos(2.0 * np.pi * fm_hz * t)

    # Corrupt the final 10 ms the way an offset ramp would.
    corrupted = pop_rate.copy()
    corrupted[int(0.19 * fs) :] = 0.0

    efr_windowed = extract_efr_amplitude(
      corrupted, sample_rate=fs, fm_hz=fm_hz, steady_state_start_s=0.05, steady_state_stop_s=0.19
    )
    efr_clean = extract_efr_amplitude(
      pop_rate, sample_rate=fs, fm_hz=fm_hz, steady_state_start_s=0.05, steady_state_stop_s=0.19
    )
    assert np.isclose(efr_windowed, efr_clean, rtol=1e-9)

  def test_stop_time_beyond_signal_is_clamped(self):
    fs = 32000
    t = np.arange(int(0.2 * fs)) / fs
    pop_rate = 5.0 + 2.0 * np.cos(2.0 * np.pi * 100.0 * t)

    efr_clamped = extract_efr_amplitude(
      pop_rate, sample_rate=fs, fm_hz=100.0, steady_state_stop_s=10.0
    )
    efr_full = extract_efr_amplitude(pop_rate, sample_rate=fs, fm_hz=100.0)
    assert np.isclose(efr_clamped, efr_full, rtol=1e-9)

  def test_invalid_parameters(self):
    valid_rate = np.ones(1000)
    with pytest.raises(ValueError, match="steady_state_stop_s"):
      extract_efr_amplitude(
        valid_rate,
        sample_rate=32000,
        fm_hz=100.0,
        steady_state_start_s=0.02,
        steady_state_stop_s=0.02,
      )
    with pytest.raises(ValueError, match="sample_rate"):
      extract_efr_amplitude(valid_rate, sample_rate=0, fm_hz=100.0)
    with pytest.raises(ValueError, match="fm_hz must be positive"):
      extract_efr_amplitude(valid_rate, sample_rate=32000, fm_hz=0.0)
    with pytest.raises(ValueError, match="Nyquist"):
      extract_efr_amplitude(valid_rate, sample_rate=32000, fm_hz=16000.0)
    with pytest.raises(ValueError, match="steady_state_start_s"):
      extract_efr_amplitude(valid_rate, sample_rate=32000, fm_hz=100.0, steady_state_start_s=-0.01)
    with pytest.raises(ValueError, match="NaN or Inf"):
      extract_efr_amplitude(np.array([1.0, np.inf]), sample_rate=32000, fm_hz=100.0)


class TestElectrophysiologyCarfacIntegration:
  """Integration tests running CARFAC outputs through electrophysiology extraction."""

  def test_carfac_click_wave_i_growth(self):
    model = build_model()
    rates = []
    for db in [40.0, 60.0, 80.0]:
      click = generate_click(duration_s=0.02, peak_db_spl=db, delay_s=0.005)
      model.reset()
      naps = model.run(click)
      pop_rate = compute_population_rate(naps)
      amp = extract_wave_i_amplitude(pop_rate, sample_rate=model.fs, stimulus_onset_s=0.005)
      rates.append(amp)

    # Wave-I amplitude should grow monotonically with click level.
    assert rates[0] > 0.0
    assert rates[1] > rates[0]
    assert rates[2] > rates[1]

  def test_carfac_sam_tone_efr_selectivity(self):
    model = build_model()
    sam = generate_sam_tone(
      fc_hz=2000.0,
      fm_hz=100.0,
      depth=1.0,
      duration_s=0.15,
      target_db_spl=70.0,
    )
    naps = model.run(sam)
    pop_rate = compute_population_rate(naps)

    # Check that response at 100 Hz dominates over off-target frequency.
    efr_100 = extract_efr_amplitude(pop_rate, sample_rate=model.fs, fm_hz=100.0)
    efr_250 = extract_efr_amplitude(pop_rate, sample_rate=model.fs, fm_hz=250.0)
    assert efr_100 > 1.0
    assert efr_100 > 3.0 * efr_250

  def test_biological_signatures_wave_i(self):
    ctrl_model = build_model(ohc_health=1.0, fiber_retention=1.0)
    syn_model = build_model(ohc_health=1.0, fiber_retention=0.5)
    ohc_model = build_model(ohc_health=0.5, fiber_retention=1.0)

    # Low-level stimulus: 40 dB SPL.
    click_low = generate_click(duration_s=0.02, peak_db_spl=40.0)
    ctrl_low = extract_wave_i_amplitude(
      compute_population_rate(ctrl_model.run(click_low)),
      sample_rate=ctrl_model.fs,
    )
    syn_low = extract_wave_i_amplitude(
      compute_population_rate(syn_model.run(click_low)),
      sample_rate=syn_model.fs,
    )
    ohc_low = extract_wave_i_amplitude(
      compute_population_rate(ohc_model.run(click_low)),
      sample_rate=ohc_model.fs,
    )

    # Synaptopathy preserves low-level response whereas OHC loss abolishes it.
    assert syn_low > 0.4 * ctrl_low
    assert ohc_low < 0.1 * ctrl_low

    # High-level stimulus: 80 dB SPL.
    ctrl_model.reset()
    syn_model.reset()
    click_high = generate_click(duration_s=0.02, peak_db_spl=80.0)
    ctrl_high = extract_wave_i_amplitude(
      compute_population_rate(ctrl_model.run(click_high)),
      sample_rate=ctrl_model.fs,
    )
    syn_high = extract_wave_i_amplitude(
      compute_population_rate(syn_model.run(click_high)),
      sample_rate=syn_model.fs,
    )

    # 50% synaptopathy causes proportional (~50%) suprathreshold reduction.
    assert syn_high < 0.7 * ctrl_high
    assert syn_high > 0.3 * ctrl_high

  def test_biological_signatures_efr(self):
    ctrl_model = build_model(ohc_health=1.0, fiber_retention=1.0)
    syn_model = build_model(ohc_health=1.0, fiber_retention=0.5)
    ohc_model = build_model(ohc_health=0.5, fiber_retention=1.0)

    # Low-level SAM tone: 40 dB SPL.
    sam_low = generate_sam_tone(duration_s=0.15, target_db_spl=40.0)
    ctrl_efr_low = extract_efr_amplitude(
      compute_population_rate(ctrl_model.run(sam_low)),
      sample_rate=ctrl_model.fs,
      fm_hz=100.0,
    )
    syn_efr_low = extract_efr_amplitude(
      compute_population_rate(syn_model.run(sam_low)),
      sample_rate=syn_model.fs,
      fm_hz=100.0,
    )
    ohc_efr_low = extract_efr_amplitude(
      compute_population_rate(ohc_model.run(sam_low)),
      sample_rate=ohc_model.fs,
      fm_hz=100.0,
    )

    # OHC loss elevates threshold (negligible response at 40 dB).
    assert syn_efr_low > 0.5 * ctrl_efr_low
    assert ohc_efr_low < 0.05 * ctrl_efr_low

    # High-level SAM tone: 80 dB SPL.
    ctrl_model.reset()
    syn_model.reset()
    sam_high = generate_sam_tone(duration_s=0.15, target_db_spl=80.0)
    ctrl_efr_high = extract_efr_amplitude(
      compute_population_rate(ctrl_model.run(sam_high)),
      sample_rate=ctrl_model.fs,
      fm_hz=100.0,
    )
    syn_efr_high = extract_efr_amplitude(
      compute_population_rate(syn_model.run(sam_high)),
      sample_rate=syn_model.fs,
      fm_hz=100.0,
    )

    # 50% synaptopathy proportionally reduces high-level EFR amplitude.
    assert syn_efr_high < 0.7 * ctrl_efr_high
    assert syn_efr_high > 0.3 * ctrl_efr_high


class TestEdgeCasesAndExports:
  """Tests for dimension handling and exports."""

  def test_2d_single_column_handling(self):
    col = np.ones((100, 1), dtype=np.float64)
    col[20] = 5.0
    amp = extract_wave_i_amplitude(col, sample_rate=32000, stimulus_onset_s=0.0)
    assert amp > 0.0

    efr = extract_efr_amplitude(col, sample_rate=32000, fm_hz=100.0)
    assert np.isfinite(efr)

  def test_invalid_dimensions_rejection(self):
    two_col = np.ones((100, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="1D"):
      extract_wave_i_amplitude(two_col, sample_rate=32000)
    with pytest.raises(ValueError, match="1D"):
      extract_efr_amplitude(two_col, sample_rate=32000, fm_hz=100.0)

  def test_string_type_rejection(self):
    with pytest.raises(TypeError):
      extract_wave_i_amplitude("not an array", sample_rate=32000)
    with pytest.raises(TypeError):
      extract_efr_amplitude("not an array", sample_rate=32000, fm_hz=100.0)

  def test_package_exports(self):
    import carfac_ephys

    assert hasattr(carfac_ephys, "compute_population_rate")
    assert hasattr(carfac_ephys, "extract_wave_i_amplitude")
    assert hasattr(carfac_ephys, "extract_efr_amplitude")
    assert hasattr(carfac_ephys, "fit_microvolts_per_au")
    assert carfac_ephys.RESPONSE_UNIT == "AU"


class TestFitMicrovoltsPerAu:
  """Tests for fit_microvolts_per_au."""

  def test_least_squares_fit_of_noisy_pairs(self):
    scale = fit_microvolts_per_au([1.0, 2.0], [0.4, 1.2])
    assert scale == pytest.approx((1.0 * 0.4 + 2.0 * 1.2) / (1.0 + 4.0))

  def test_rejects_mismatched_lengths(self):
    with pytest.raises(ValueError, match="equally long"):
      fit_microvolts_per_au([1.0, 2.0], [1.0])

  def test_rejects_empty_inputs(self):
    with pytest.raises(ValueError, match="empty"):
      fit_microvolts_per_au([], [])

  def test_rejects_zero_responses(self):
    with pytest.raises(ValueError, match="non-zero"):
      fit_microvolts_per_au([0.0, 0.0], [1.0, 2.0])

  def test_rejects_non_positive_scale(self):
    with pytest.raises(ValueError, match="positive"):
      fit_microvolts_per_au([1.0], [-1.0])

  def test_rejects_non_finite_values(self):
    with pytest.raises(ValueError, match="NaN or Inf"):
      fit_microvolts_per_au([1.0, np.nan], [1.0, 1.0])
