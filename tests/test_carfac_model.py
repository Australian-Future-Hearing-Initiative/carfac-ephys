"""Tests for CARFAC model wrapper and impairment configurations."""

import numpy as np
import pytest

from carfac_ephys.carfac_model import (
  CarfacModel,
  FiberRetention,
  build_model,
)
from carfac_ephys.constants import DEFAULT_SAMPLE_RATE
from carfac_ephys.stimuli import generate_click, generate_sam_tone


class TestFiberRetention:
  """Tests for FiberRetention named tuple."""

  def test_defaults_and_custom(self):
    ret = FiberRetention()
    assert ret.hsr == 1.0 and ret.msr == 1.0 and ret.lsr == 1.0
    assert tuple(ret) == (1.0, 1.0, 1.0)

    custom = FiberRetention(hsr=0.8, msr=0.5, lsr=0.2)
    assert custom.hsr == 0.8 and custom.msr == 0.5 and custom.lsr == 0.2
    assert tuple(custom) == (0.8, 0.5, 0.2)


class TestBuildModelInit:
  """Tests for CARFAC model construction and default parameters."""

  def test_default_initialization(self):
    model = build_model()
    assert isinstance(model, CarfacModel)
    assert model.fs == DEFAULT_SAMPLE_RATE
    assert model.n_channels == 77
    assert model.pole_freqs.shape == (77,)
    assert model.pole_freqs[0] > model.pole_freqs[-1]

    # Verify CARFAC design parameters.
    ear_params = model.params.ears[0]
    assert ear_params.ihc.ihc_style == "two_cap_with_syn"
    assert ear_params.car.use_delay_buffer is True

    # Verify default healthy OHC and fiber states.
    assert np.allclose(model.ohc_health, 1.0)
    assert model.n_fibers.shape == (77, 3)
    assert np.allclose(model.n_fibers[:, 0], 500.0)
    assert np.allclose(model.n_fibers[:, 1], 350.0)
    assert np.allclose(model.n_fibers[:, 2], 250.0)

  def test_custom_sampling_rate(self):
    fs = 16000
    model = build_model(fs=fs)
    assert model.fs == fs
    assert model.n_channels > 0
    assert model.pole_freqs.shape == (model.n_channels,)

  def test_invalid_sampling_rate(self):
    with pytest.raises(ValueError):
      build_model(fs=0)
    with pytest.raises(ValueError):
      build_model(fs=-16000)


class TestOhcHealth:
  """Tests for outer hair cell health configuration and validation."""

  def test_scalar_health(self):
    for health in [0.5, 0.0]:
      model = build_model(ohc_health=health)
      assert model.ohc_health.shape == (77,)
      assert np.allclose(model.ohc_health, health)

  def test_array_health(self):
    health = np.linspace(1.0, 0.2, 77, dtype=np.float32)
    model = build_model(ohc_health=health)
    assert np.allclose(model.ohc_health, health)

  def test_sequence_health(self):
    health_list = [0.8] * 77
    model = build_model(ohc_health=health_list)
    assert np.allclose(model.ohc_health, 0.8)

  def test_invalid_health_bounds(self):
    with pytest.raises(ValueError):
      build_model(ohc_health=-0.1)
    with pytest.raises(ValueError):
      build_model(ohc_health=1.05)

    # Array with negative element.
    bad_arr = np.ones(77, dtype=np.float32)
    bad_arr[10] = -0.01
    with pytest.raises(ValueError):
      build_model(ohc_health=bad_arr)

    # Array with exceeding element.
    bad_arr2 = np.ones(77, dtype=np.float32)
    bad_arr2[10] = 1.1
    with pytest.raises(ValueError):
      build_model(ohc_health=bad_arr2)

    # Array with non-finite element.
    bad_arr_nan = np.ones(77, dtype=np.float32)
    bad_arr_nan[10] = np.nan
    with pytest.raises(ValueError):
      build_model(ohc_health=bad_arr_nan)

  def test_invalid_health_dimensions(self):
    # Length mismatch.
    with pytest.raises(ValueError):
      build_model(ohc_health=np.ones(50))

    # 2D array.
    with pytest.raises(ValueError):
      build_model(ohc_health=np.ones((77, 2)))

  def test_numpy_scalar_health(self):
    # pyrefly: ignore [bad-argument-type]
    model_f32 = build_model(ohc_health=np.float32(0.5))
    assert np.allclose(model_f32.ohc_health, 0.5)
    model_f64 = build_model(ohc_health=np.float64(0.8))
    assert np.allclose(model_f64.ohc_health, 0.8)

  def test_invalid_health_type(self):
    with pytest.raises(TypeError):
      build_model(ohc_health="healthy")  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(TypeError):
      build_model(ohc_health=True)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(TypeError):
      build_model(ohc_health=["healthy"] * 77)  # pyrefly: ignore[bad-argument-type]


class TestFiberRetentionScaling:
  """Tests for fiber retention scaling across fiber types."""

  def test_scalar_retention(self):
    for ret, expected_hsr in [(0.5, 250.0), (0.0, 0.0)]:
      model = build_model(fiber_retention=ret)
      assert np.allclose(model.n_fibers[:, 0], expected_hsr)

  def test_sequence_retention(self):
    ret = FiberRetention(hsr=0.6, msr=0.4, lsr=0.2)
    model = build_model(fiber_retention=ret)
    assert np.allclose(model.n_fibers[:, 0], 500.0 * 0.6)
    assert np.allclose(model.n_fibers[:, 1], 350.0 * 0.4)
    assert np.allclose(model.n_fibers[:, 2], 250.0 * 0.2)

  def test_array_retention(self):
    arr = np.array([0.7, 0.6, 0.5], dtype=np.float32)
    model = build_model(fiber_retention=arr)
    assert np.allclose(model.n_fibers[:, 0], 500.0 * 0.7)
    assert np.allclose(model.n_fibers[:, 1], 350.0 * 0.6)
    assert np.allclose(model.n_fibers[:, 2], 250.0 * 0.5)

  def test_invalid_fiber_bounds(self):
    with pytest.raises(ValueError):
      build_model(fiber_retention=-0.1)
    with pytest.raises(ValueError):
      build_model(fiber_retention=1.2)
    with pytest.raises(ValueError):
      build_model(fiber_retention=(-0.1, 1.0, 1.0))
    with pytest.raises(ValueError):
      build_model(fiber_retention=(1.0, 1.5, 1.0))
    with pytest.raises(ValueError):
      build_model(fiber_retention=np.nan)
    with pytest.raises(ValueError):
      build_model(fiber_retention=(np.nan, 1.0, 1.0))
    with pytest.raises(ValueError):
      build_model(fiber_retention=(np.inf, 1.0, 1.0))

  def test_invalid_fiber_length(self):
    with pytest.raises(ValueError):
      build_model(fiber_retention=(0.5, 0.5))
    with pytest.raises(ValueError):
      build_model(fiber_retention=(0.5, 0.5, 0.5, 0.5))

  def test_numpy_scalar_retention(self):
    # pyrefly: ignore [bad-argument-type]
    model_f32 = build_model(fiber_retention=np.float32(0.5))
    assert np.allclose(model_f32.n_fibers[:, 0], 250.0)
    model_f64 = build_model(fiber_retention=np.float64(0.8))
    assert np.allclose(model_f64.n_fibers[:, 0], 400.0)

  def test_invalid_fiber_type(self):
    with pytest.raises(TypeError):
      build_model(fiber_retention="full")  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(TypeError):
      build_model(fiber_retention=False)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(TypeError):
      build_model(fiber_retention=("a", "b", "c"))  # pyrefly: ignore[bad-argument-type]


class TestCarfacModelRun:
  """Tests for forward execution and NAPs generation."""

  def test_run_click_stimulus(self):
    model = build_model()
    click = generate_click(duration_s=0.01, peak_db_spl=80.0)
    naps = model.run(click)

    # Check output shape and absence of NaNs.
    assert naps.shape == (len(click), model.n_channels)
    assert naps.dtype == np.float32
    assert not np.isnan(naps).any()
    assert not np.isinf(naps).any()
    assert np.max(np.abs(naps)) > 0.0

  def test_run_sam_tone_stimulus(self):
    model = build_model()
    sam = generate_sam_tone(
      fc_hz=2000.0,
      fm_hz=100.0,
      duration_s=0.02,
      target_db_spl=70.0,
    )
    naps = model.run(sam)

    assert naps.shape == (len(sam), model.n_channels)
    assert not np.isnan(naps).any()
    assert not np.isinf(naps).any()

  def test_run_2d_single_channel_waveform(self):
    model = build_model()
    click = generate_click(duration_s=0.01, delay_s=0.002, peak_db_spl=70.0)
    click_2d = click[:, np.newaxis]
    naps = model.run(click_2d)
    assert naps.shape == (len(click), model.n_channels)

  def test_run_zero_sample_waveform(self):
    model = build_model()
    empty = np.zeros(0, dtype=np.float32)
    naps = model.run(empty)
    assert naps.shape == (0, model.n_channels)

  def test_run_invalid_waveform_inputs(self):
    model = build_model()

    # Multichannel 2D input.
    with pytest.raises(ValueError):
      model.run(np.zeros((100, 2), dtype=np.float32))

    # 3D input.
    with pytest.raises(ValueError):
      model.run(np.zeros((100, 1, 1), dtype=np.float32))

    # Input containing NaNs.
    nan_wave = np.zeros(100, dtype=np.float32)
    nan_wave[10] = np.nan
    with pytest.raises(ValueError):
      model.run(nan_wave)

    # Input containing Infs.
    inf_wave = np.zeros(100, dtype=np.float32)
    inf_wave[10] = np.inf
    with pytest.raises(ValueError):
      model.run(inf_wave)

  def test_reset_restores_initial_state(self):
    model = build_model()
    click = generate_click(duration_s=0.01, peak_db_spl=80.0)

    # Run click twice: state changes from initial.
    naps1 = model.run(click)
    naps2 = model.run(click)
    # State has progressed so responses differ slightly.
    assert not np.allclose(naps1, naps2)

    # Resetting should restore exact state for repeating segment 1.
    model.reset()
    naps_reset = model.run(click)
    assert np.allclose(naps1, naps_reset)


class TestBiologicalSignatures:
  """Tests verifying biophysical impairment impacts on response amplitudes."""

  def test_ohc_loss_reduces_response(self):
    healthy_model = build_model(ohc_health=1.0)
    impaired_model = build_model(ohc_health=0.2)

    stim = generate_click(duration_s=0.015, peak_db_spl=80.0)
    healthy_naps = healthy_model.run(stim)
    impaired_naps = impaired_model.run(stim)

    healthy_amp = np.ptp(healthy_naps.mean(axis=1))
    impaired_amp = np.ptp(impaired_naps.mean(axis=1))
    assert impaired_amp < 0.1 * healthy_amp

  def test_synaptopathy_reduces_response_proportionally(self):
    healthy_model = build_model(fiber_retention=1.0)
    syn_model = build_model(fiber_retention=0.5)

    stim = generate_click(duration_s=0.015, peak_db_spl=80.0)
    healthy_naps = healthy_model.run(stim)
    syn_naps = syn_model.run(stim)

    healthy_amp = np.ptp(healthy_naps.mean(axis=1))
    syn_amp = np.ptp(syn_naps.mean(axis=1))
    # 50% fiber retention should roughly halve the response.
    assert syn_amp < 0.7 * healthy_amp
    assert syn_amp > 0.3 * healthy_amp


class TestPackageExports:
  """Tests that carfac_model exports are accessible from root package."""

  def test_exports(self):
    import carfac_ephys

    assert hasattr(carfac_ephys, "CarfacModel")
    assert hasattr(carfac_ephys, "FiberRetention")
    assert hasattr(carfac_ephys, "build_model")
