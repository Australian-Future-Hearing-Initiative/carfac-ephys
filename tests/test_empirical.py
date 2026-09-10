"""Tests for the empirical chinchilla ABR dataset loader."""

import json
import pathlib

import pytest

from carfac_ephys.empirical import (
  CLICK_FREQUENCY_HZ,
  DEFAULT_DATA_DIR,
  PER_ANIMAL_FILE_NAME,
  SUMMARY_FILE_NAME,
  AnimalWaveAmplitudes,
  PrePostStat,
  load_chinchilla_abr_dataset,
)


@pytest.fixture(name="dataset", scope="module")
def dataset_fixture():
  """Loads the shipped dataset once for all tests."""
  return load_chinchilla_abr_dataset()


def _write_dataset(directory: pathlib.Path, summary: dict, csv_rows: list[str]) -> pathlib.Path:
  """Writes a synthetic dataset into a directory and returns it."""
  (directory / SUMMARY_FILE_NAME).write_text(json.dumps(summary), encoding="utf-8")
  header = "ID,TimePoint,W1,W5\n"
  (directory / PER_ANIMAL_FILE_NAME).write_text(
    header + "\n".join(csv_rows) + "\n", encoding="utf-8"
  )
  return directory


def _minimal_summary() -> dict:
  """Returns a minimal well-formed summary structure with two frequencies."""
  waves = {
    "mean_pre": [2.0, 1.0, 4.0],
    "mean_post": [1.0, 0.5, 2.0],
    "std_pre": [0.2, 0.1, 0.4],
    "std_post": [0.1, 0.05, 0.2],
  }
  return {
    "source": "synthetic",
    "animals": ["A1"],
    "frequencies_hz": [0, 4000],
    "thresholds_db_spl": {
      "mean_pre": [10.0, 20.0],
      "mean_post": [12.0, 25.0],
      "std_pre": [1.0, 2.0],
      "std_post": [1.5, 2.5],
    },
    "high_level_w1_uv": dict(waves),
    "high_level_w5_uv": dict(waves),
  }


class TestPrePostStat:
  """Tests for PrePostStat derived quantities."""

  def test_shift_and_ratio(self):
    stat = PrePostStat(mean_pre=2.0, mean_post=1.5, std_pre=0.1, std_post=0.2)
    assert stat.shift == pytest.approx(-0.5)
    assert stat.ratio == pytest.approx(0.75)

  def test_ratio_rejects_zero_baseline(self):
    stat = PrePostStat(mean_pre=0.0, mean_post=1.0, std_pre=0.0, std_post=0.0)
    with pytest.raises(ZeroDivisionError):
      _ = stat.ratio


class TestAnimalWaveAmplitudes:
  """Tests for per-animal amplitude ratios."""

  def test_wave_ratios(self):
    animal = AnimalWaveAmplitudes(
      animal_id="Q1", pre_w1_uv=2.0, post_w1_uv=1.0, pre_w5_uv=4.0, post_w5_uv=1.0
    )
    assert animal.w1_ratio == pytest.approx(0.5)
    assert animal.w5_ratio == pytest.approx(0.25)

  def test_ratio_rejects_zero_baseline(self):
    animal = AnimalWaveAmplitudes(
      animal_id="Q1", pre_w1_uv=0.0, post_w1_uv=1.0, pre_w5_uv=0.0, post_w5_uv=1.0
    )
    with pytest.raises(ZeroDivisionError):
      _ = animal.w1_ratio
    with pytest.raises(ZeroDivisionError):
      _ = animal.w5_ratio


class TestLoadShippedDataset:
  """Tests against the dataset shipped in the repository data directory."""

  def test_data_dir_exists(self):
    assert (DEFAULT_DATA_DIR / SUMMARY_FILE_NAME).is_file()
    assert (DEFAULT_DATA_DIR / PER_ANIMAL_FILE_NAME).is_file()

  def test_source_and_animals(self, dataset):
    assert "Bharadwaj" in dataset.source
    assert len(dataset.animals) == 7
    assert dataset.animals[0] == "Q348"

  def test_frequencies_include_click(self, dataset):
    assert dataset.frequencies_hz == (0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    assert CLICK_FREQUENCY_HZ in dataset.thresholds_db_spl

  def test_click_threshold_is_preserved(self, dataset):
    # Temporary threshold shift protocol: click threshold recovers by two weeks.
    assert abs(dataset.click_threshold_shift_db) <= 3.0
    assert dataset.thresholds_db_spl[0.0].mean_pre == pytest.approx(18.7749, abs=1e-3)

  def test_tone_thresholds_shift_modestly(self, dataset):
    shift_4k = dataset.threshold_shift_db(4000.0)
    assert 0.0 < shift_4k <= 5.0

  def test_suprathreshold_wave_i_attenuation(self, dataset):
    # Wave-I is reduced at high level despite recovered thresholds.
    assert dataset.suprathreshold_w1_ratio == pytest.approx(0.717, abs=0.01)
    assert dataset.wave_i_ratio(CLICK_FREQUENCY_HZ) < 1.0

  def test_wave_v_is_less_affected_than_wave_i(self, dataset):
    assert dataset.tone_average_w5_uv.ratio > dataset.tone_average_w1_uv.ratio

  def test_per_animal_matches_summary_animals(self, dataset):
    assert tuple(animal.animal_id for animal in dataset.per_animal) == dataset.animals

  def test_per_animal_ratios(self, dataset):
    ratios = dataset.per_animal_w1_ratios
    assert len(ratios) == 7
    assert all(ratio > 0.0 for ratio in ratios)

  def test_per_animal_values_match_csv(self, dataset):
    q348 = dataset.per_animal[0]
    assert q348.pre_w1_uv == pytest.approx(1.04276378865147)
    assert q348.post_w1_uv == pytest.approx(0.631910633176243)

  def test_unknown_frequency_raises(self, dataset):
    with pytest.raises(KeyError, match="available frequencies"):
      dataset.threshold_shift_db(12345.0)


class TestLoadSyntheticDataset:
  """Tests for loader behaviour on synthetic inputs."""

  def test_loads_from_custom_directory(self, tmp_path):
    _write_dataset(tmp_path, _minimal_summary(), ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    dataset = load_chinchilla_abr_dataset(tmp_path)
    assert dataset.source == "synthetic"
    assert dataset.tone_average_w1_uv.ratio == pytest.approx(0.5)
    assert dataset.per_animal_w1_ratios == pytest.approx((0.5,))

  def test_accepts_string_path(self, tmp_path):
    _write_dataset(tmp_path, _minimal_summary(), ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    assert load_chinchilla_abr_dataset(str(tmp_path)).animals == ("A1",)

  def test_missing_files_raise(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      load_chinchilla_abr_dataset(tmp_path)

  def test_mismatched_animals_raise(self, tmp_path):
    _write_dataset(tmp_path, _minimal_summary(), ["B9,pre,2.0,4.0", "B9,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="Animal identifiers disagree"):
      load_chinchilla_abr_dataset(tmp_path)

  def test_missing_time_point_raises(self, tmp_path):
    _write_dataset(tmp_path, _minimal_summary(), ["A1,pre,2.0,4.0"])
    with pytest.raises(ValueError, match="missing time points"):
      load_chinchilla_abr_dataset(tmp_path)

  def test_missing_series_raises(self, tmp_path):
    summary = _minimal_summary()
    del summary["thresholds_db_spl"]["std_post"]
    _write_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="missing series"):
      load_chinchilla_abr_dataset(tmp_path)

  def test_inconsistent_series_lengths_raise(self, tmp_path):
    summary = _minimal_summary()
    summary["thresholds_db_spl"]["std_post"] = [1.0]
    _write_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="differing lengths"):
      load_chinchilla_abr_dataset(tmp_path)

  def test_unexpected_entry_count_raises(self, tmp_path):
    summary = _minimal_summary()
    for key in summary["high_level_w1_uv"]:
      summary["high_level_w1_uv"][key] = [1.0]
    _write_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="expected"):
      load_chinchilla_abr_dataset(tmp_path)

  def test_missing_tone_average_raises(self, tmp_path):
    summary = _minimal_summary()
    for block in ("high_level_w1_uv", "high_level_w5_uv"):
      for key in summary[block]:
        summary[block][key] = summary[block][key][:2]
    _write_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="trailing 4/8 kHz average"):
      load_chinchilla_abr_dataset(tmp_path)

  def test_empty_csv_raises(self, tmp_path):
    summary = _minimal_summary()
    (tmp_path / SUMMARY_FILE_NAME).write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / PER_ANIMAL_FILE_NAME).write_text("ID,TimePoint,W1,W5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No rows found"):
      load_chinchilla_abr_dataset(tmp_path)
