"""Tests for the empirical ABR dataset loader, across both study designs."""

import dataclasses
import json
import pathlib

import pytest

from carfac_ephys.empirical import (
  CLICK_FREQUENCY_HZ,
  DEFAULT_DATA_DIR,
  INDEPENDENT_GROUPS,
  PAIRED_TIMEPOINTS,
  SPECIES_DATA_FILES,
  AbrDataset,
  GroupComparisonStat,
  SubjectWaveAmplitudes,
  load_abr_dataset,
)

CHINCHILLA_FILES = SPECIES_DATA_FILES["chinchilla"]
HUMAN_FILES = SPECIES_DATA_FILES["human"]
# A paired design always names a per-subject file. Binding it once narrows the
# optional type for the paired tests below.
CHINCHILLA_PER_SUBJECT = CHINCHILLA_FILES.per_subject_file_name
assert CHINCHILLA_PER_SUBJECT is not None


@pytest.fixture(name="chinchilla", scope="module")
def chinchilla_fixture():
  """Loads the shipped chinchilla dataset once for all tests."""
  return load_abr_dataset("chinchilla")


@pytest.fixture(name="human", scope="module")
def human_fixture():
  """Loads the shipped human dataset once for all tests."""
  return load_abr_dataset("human")


def _write_paired_dataset(
  directory: pathlib.Path, summary: dict, csv_rows: list[str]
) -> pathlib.Path:
  """Writes a synthetic paired (chinchilla-shaped) dataset into a directory."""
  (directory / CHINCHILLA_FILES.summary_file_name).write_text(json.dumps(summary), encoding="utf-8")
  header = "ID,TimePoint,W1,W5\n"
  (directory / CHINCHILLA_PER_SUBJECT).write_text(
    header + "\n".join(csv_rows) + "\n", encoding="utf-8"
  )
  return directory


def _minimal_paired_summary() -> dict:
  """Returns a minimal well-formed paired summary with two frequencies."""
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


def _write_grouped_dataset(directory: pathlib.Path, summary: dict) -> pathlib.Path:
  """Writes a synthetic independent-groups (human-shaped) summary."""
  (directory / HUMAN_FILES.summary_file_name).write_text(json.dumps(summary), encoding="utf-8")
  return directory


def _minimal_grouped_summary() -> dict:
  """Returns a minimal well-formed independent-groups summary.

  The two comparison groups carry deliberately different means so a test can
  tell whether the requested group was actually the one used.
  """

  def measure(mean: float, std: float, n: int) -> dict:
    return {"mean": mean, "std": std, "n": n}

  return {
    "source": "synthetic",
    "design": "independent_groups",
    "baseline_group": "ctrl",
    "comparison_groups": ["nexp", "ma"],
    "subjects": ["S1", "S2", "S3"],
    "measures": {
      "wave1_uv": {"units": "uV", "description": "Wave-I."},
      "wave5_uv": {"units": "uV", "description": "Wave-V."},
      "audiometric_lfa_db_hl": {"units": "dB HL", "description": "Behavioural PTA."},
    },
    "groups": {
      "ctrl": {
        "n": 10,
        "wave1_uv": measure(2.0, 0.4, 10),
        "wave5_uv": measure(1.0, 0.2, 10),
        "audiometric_lfa_db_hl": measure(5.0, 1.0, 10),
      },
      "nexp": {
        "n": 8,
        "wave1_uv": measure(1.0, 0.3, 7),
        "wave5_uv": measure(0.8, 0.1, 7),
        "audiometric_lfa_db_hl": measure(6.0, 1.5, 8),
      },
      "ma": {
        "n": 9,
        "wave1_uv": measure(0.5, 0.2, 9),
        "wave5_uv": measure(0.4, 0.1, 9),
        "audiometric_lfa_db_hl": measure(9.0, 2.0, 9),
      },
    },
  }


class TestGroupComparisonStat:
  """Tests for the neutral group-level statistic."""

  def test_difference_and_ratio(self):
    stat = GroupComparisonStat(
      mean_baseline=2.0, mean_comparison=1.5, std_baseline=0.1, std_comparison=0.2
    )
    assert stat.difference == pytest.approx(-0.5)
    assert stat.ratio_of_means == pytest.approx(0.75)

  def test_ratio_rejects_zero_baseline(self):
    stat = GroupComparisonStat(
      mean_baseline=0.0, mean_comparison=1.0, std_baseline=0.0, std_comparison=0.0
    )
    with pytest.raises(ZeroDivisionError):
      _ = stat.ratio_of_means


class TestSubjectWaveAmplitudes:
  """Tests for per-subject ratios, which exist only in a paired design."""

  def test_wave_ratios(self):
    subject = SubjectWaveAmplitudes(
      subject_id="Q1",
      baseline_w1_uv=2.0,
      comparison_w1_uv=1.0,
      baseline_w5_uv=4.0,
      comparison_w5_uv=1.0,
    )
    assert subject.w1_ratio == pytest.approx(0.5)
    assert subject.w5_ratio == pytest.approx(0.25)

  def test_ratio_rejects_zero_baseline(self):
    subject = SubjectWaveAmplitudes(
      subject_id="Q1",
      baseline_w1_uv=0.0,
      comparison_w1_uv=1.0,
      baseline_w5_uv=0.0,
      comparison_w5_uv=1.0,
    )
    with pytest.raises(ZeroDivisionError):
      _ = subject.w1_ratio
    with pytest.raises(ZeroDivisionError):
      _ = subject.w5_ratio


class TestSpeciesRegistry:
  """The registry records each dataset's design and comparison axis."""

  def test_chinchilla_is_paired_on_timepoint(self):
    assert CHINCHILLA_FILES.design == PAIRED_TIMEPOINTS
    assert CHINCHILLA_FILES.condition_column == "TimePoint"
    assert CHINCHILLA_FILES.baseline_label == "pre"
    assert CHINCHILLA_FILES.comparison_labels == ("2wk",)

  def test_human_is_independent_groups(self):
    assert HUMAN_FILES.design == INDEPENDENT_GROUPS
    assert HUMAN_FILES.condition_column == "Group"
    assert HUMAN_FILES.baseline_label == "ctrl"
    assert HUMAN_FILES.comparison_labels == ("nexp", "ma")

  def test_unsupported_species_raises(self, tmp_path):
    with pytest.raises(ValueError, match="Unsupported species"):
      load_abr_dataset("axolotl", tmp_path)


class TestShippedChinchillaDataset:
  """Tests against the chinchilla dataset shipped in the repository."""

  def test_data_files_exist(self):
    assert (DEFAULT_DATA_DIR / CHINCHILLA_FILES.summary_file_name).is_file()
    assert (DEFAULT_DATA_DIR / CHINCHILLA_PER_SUBJECT).is_file()

  def test_species_design_and_source(self, chinchilla):
    assert chinchilla.species == "chinchilla"
    assert chinchilla.design == PAIRED_TIMEPOINTS
    assert chinchilla.is_paired is True
    assert "Bharadwaj" in chinchilla.source

  def test_subjects(self, chinchilla):
    assert len(chinchilla.subjects) == 7
    assert chinchilla.subjects[0] == "Q348"

  def test_frequencies_include_click(self, chinchilla):
    assert chinchilla.frequencies_hz == (0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    assert chinchilla.has_abr_thresholds is True
    assert CLICK_FREQUENCY_HZ in chinchilla.abr_thresholds_db_spl

  def test_click_threshold_is_preserved(self, chinchilla):
    # Temporary threshold shift protocol: click threshold recovers by two weeks.
    assert chinchilla.click_threshold_shift_db == pytest.approx(-0.3024, abs=1e-3)
    assert chinchilla.abr_thresholds_db_spl[0.0].mean_baseline == pytest.approx(18.7749, abs=1e-3)

  def test_tone_thresholds_shift_modestly(self, chinchilla):
    assert 0.0 < chinchilla.threshold_shift_db(4000.0) <= 5.0

  def test_suprathreshold_wave_i_attenuation(self, chinchilla):
    assert chinchilla.wave_i_ratio(CLICK_FREQUENCY_HZ) == pytest.approx(0.742, abs=0.01)
    assert chinchilla.ratio_of_mean_w1 == pytest.approx(0.717, abs=0.01)

  def test_reference_ratio_is_the_click_condition_not_the_tone_average(self, chinchilla):
    # The simulation is click-evoked, so the comparison uses the click ratio;
    # ratio_of_mean_w1 stays the tone-average quantity.
    assert chinchilla.reference_w1_ratio == pytest.approx(0.742, abs=0.01)
    assert chinchilla.reference_w1_ratio != pytest.approx(chinchilla.ratio_of_mean_w1)

  def test_wave_v_is_less_affected_than_wave_i(self, chinchilla):
    assert chinchilla.has_wave_v is True
    assert chinchilla.wave5_uv.ratio_of_means > chinchilla.wave1_uv.ratio_of_means

  def test_per_subject_ratios_exist_and_match_subjects(self, chinchilla):
    assert tuple(s.subject_id for s in chinchilla.per_subject) == chinchilla.subjects
    assert len(chinchilla.paired_w1_ratios) == 7
    assert all(ratio > 0.0 for ratio in chinchilla.paired_w1_ratios)

  def test_per_subject_values_match_csv(self, chinchilla):
    q348 = chinchilla.per_subject[0]
    assert q348.baseline_w1_uv == pytest.approx(1.04276378865147)
    assert q348.comparison_w1_uv == pytest.approx(0.631910633176243)

  def test_calibration_reference_is_the_baseline_click_amplitude(self, chinchilla):
    assert chinchilla.baseline_reference_w1_uv == pytest.approx(
      chinchilla.frequency_wave1_uv[CLICK_FREQUENCY_HZ].mean_baseline
    )

  def test_unknown_frequency_raises(self, chinchilla):
    with pytest.raises(KeyError, match="available frequencies"):
      chinchilla.threshold_shift_db(12345.0)


class TestShippedHumanDataset:
  """Tests against the human dataset shipped in the repository.

  These pin the values actually shipped, so a change to the CSV or the
  generator that moves a group mean fails here rather than silently.
  """

  def test_data_file_exists(self):
    assert (DEFAULT_DATA_DIR / HUMAN_FILES.summary_file_name).is_file()

  def test_species_design_and_default_group(self, human):
    assert human.species == "human"
    assert human.design == INDEPENDENT_GROUPS
    assert human.is_paired is False
    assert human.baseline_label == "ctrl"
    assert human.comparison_group == "nexp"

  def test_reports_no_abr_thresholds(self, human):
    # Behavioural dB HL audiometry is not an ABR threshold in dB SPL, so the
    # dataset must not expose one.
    assert human.has_abr_thresholds is False
    assert human.abr_thresholds_db_spl is None
    with pytest.raises(ValueError, match="no ABR thresholds"):
      _ = human.click_threshold_shift_db

  def test_has_no_frequency_resolved_waves(self, human):
    assert human.frequency_wave1_uv is None
    with pytest.raises(ValueError, match="not resolved by stimulus frequency"):
      _ = human.wave_i_ratio()

  def test_no_per_subject_ratios_in_an_unpaired_design(self, human):
    assert human.per_subject == ()
    assert human.paired_w1_ratios == ()

  def test_group_sizes_and_wave_i_means(self, human):
    assert human.wave1_uv.n_baseline == 54
    assert human.wave1_uv.n_comparison == 53
    assert human.wave1_uv.mean_baseline == pytest.approx(1.1622, abs=1e-4)
    assert human.wave1_uv.mean_comparison == pytest.approx(1.0972, abs=1e-4)

  def test_ratio_of_mean_wave_i(self, human):
    assert human.ratio_of_mean_w1 == pytest.approx(0.9441, abs=1e-4)

  def test_selecting_the_other_group_changes_the_statistics(self):
    ma = load_abr_dataset("human", comparison_group="ma")
    assert ma.comparison_group == "ma"
    assert ma.wave1_uv.mean_comparison == pytest.approx(0.6137, abs=1e-4)
    assert ma.ratio_of_mean_w1 == pytest.approx(0.5280, abs=1e-4)

  def test_invalid_comparison_group_raises(self):
    with pytest.raises(ValueError, match="is not valid for this dataset"):
      load_abr_dataset("human", comparison_group="bogus")

  def test_audiometry_is_context_carrying_its_own_units(self, human):
    lfa = human.context_measures["audiometric_lfa_db_hl"]
    assert lfa.units == "dB HL"
    assert lfa.mean_baseline == pytest.approx(5.1136, abs=1e-4)
    assert human.wave1_uv.units == "uV"

  def test_reference_ratio_falls_back_to_the_group_mean(self, human):
    assert human.reference_w1_ratio == pytest.approx(human.ratio_of_mean_w1)

  def test_calibration_reference_falls_back_to_the_group_mean(self, human):
    assert human.baseline_reference_w1_uv == pytest.approx(human.wave1_uv.mean_baseline)


class TestSyntheticPairedDataset:
  """Loader behaviour on synthetic paired inputs."""

  def test_loads_from_custom_directory(self, tmp_path):
    _write_paired_dataset(tmp_path, _minimal_paired_summary(), ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    dataset = load_abr_dataset("chinchilla", tmp_path)
    assert isinstance(dataset, AbrDataset)
    assert dataset.source == "synthetic"
    assert dataset.ratio_of_mean_w1 == pytest.approx(0.5)
    assert dataset.paired_w1_ratios == pytest.approx((0.5,))

  def test_accepts_string_path(self, tmp_path):
    _write_paired_dataset(tmp_path, _minimal_paired_summary(), ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    assert load_abr_dataset("chinchilla", str(tmp_path)).subjects == ("A1",)

  def test_missing_files_raise(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      load_abr_dataset("chinchilla", tmp_path)

  def test_mismatched_subjects_raise(self, tmp_path):
    _write_paired_dataset(tmp_path, _minimal_paired_summary(), ["B9,pre,2.0,4.0", "B9,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="Subject identifiers disagree"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_missing_condition_raises(self, tmp_path):
    _write_paired_dataset(tmp_path, _minimal_paired_summary(), ["A1,pre,2.0,4.0"])
    with pytest.raises(ValueError, match="missing conditions"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_missing_series_raises(self, tmp_path):
    summary = _minimal_paired_summary()
    del summary["thresholds_db_spl"]["std_post"]
    _write_paired_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="missing series"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_inconsistent_series_lengths_raise(self, tmp_path):
    summary = _minimal_paired_summary()
    summary["thresholds_db_spl"]["std_post"] = [1.0]
    _write_paired_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="differing lengths"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_unexpected_entry_count_raises(self, tmp_path):
    summary = _minimal_paired_summary()
    for key in summary["high_level_w1_uv"]:
      summary["high_level_w1_uv"][key] = [1.0]
    _write_paired_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="expected"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_missing_tone_average_raises(self, tmp_path):
    summary = _minimal_paired_summary()
    for block in ("high_level_w1_uv", "high_level_w5_uv"):
      for key in summary[block]:
        summary[block][key] = summary[block][key][:2]
    _write_paired_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="trailing tone-average entry"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_empty_csv_raises(self, tmp_path):
    _write_paired_dataset(tmp_path, _minimal_paired_summary(), [])
    with pytest.raises(ValueError, match="No rows found"):
      load_abr_dataset("chinchilla", tmp_path)


class TestSyntheticGroupedDataset:
  """Loader behaviour on synthetic independent-groups inputs."""

  def test_defaults_to_the_first_comparison_label(self, tmp_path):
    _write_grouped_dataset(tmp_path, _minimal_grouped_summary())
    dataset = load_abr_dataset("human", tmp_path)
    assert dataset.comparison_group == "nexp"
    assert dataset.ratio_of_mean_w1 == pytest.approx(0.5)

  def test_requested_group_selects_its_own_statistics(self, tmp_path):
    _write_grouped_dataset(tmp_path, _minimal_grouped_summary())
    dataset = load_abr_dataset("human", tmp_path, comparison_group="ma")
    assert dataset.ratio_of_mean_w1 == pytest.approx(0.25)
    assert dataset.wave1_uv.n_comparison == 9

  def test_context_measures_keep_their_units(self, tmp_path):
    _write_grouped_dataset(tmp_path, _minimal_grouped_summary())
    dataset = load_abr_dataset("human", tmp_path)
    assert set(dataset.context_measures) == {"audiometric_lfa_db_hl"}
    assert dataset.context_measures["audiometric_lfa_db_hl"].units == "dB HL"

  def test_no_per_subject_file_is_required(self, tmp_path):
    # An independent-groups dataset is fully described by its group summary.
    _write_grouped_dataset(tmp_path, _minimal_grouped_summary())
    assert HUMAN_FILES.per_subject_file_name is None
    assert list(tmp_path.glob("*.csv")) == []
    assert load_abr_dataset("human", tmp_path).per_subject == ()

  def test_missing_summary_raises(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      load_abr_dataset("human", tmp_path)

  def test_unknown_group_raises(self, tmp_path):
    summary = _minimal_grouped_summary()
    del summary["groups"]["ma"]
    _write_grouped_dataset(tmp_path, summary)
    with pytest.raises(ValueError, match="no 'ma' group"):
      load_abr_dataset("human", tmp_path, comparison_group="ma")

  def test_missing_measure_raises(self, tmp_path):
    summary = _minimal_grouped_summary()
    del summary["groups"]["nexp"]["wave1_uv"]
    _write_grouped_dataset(tmp_path, summary)
    with pytest.raises(ValueError, match="no 'wave1_uv' measure"):
      load_abr_dataset("human", tmp_path)

  def test_wave_v_is_optional(self, tmp_path):
    summary = _minimal_grouped_summary()
    for group in summary["groups"].values():
      del group["wave5_uv"]
    del summary["measures"]["wave5_uv"]
    _write_grouped_dataset(tmp_path, summary)
    dataset = load_abr_dataset("human", tmp_path)
    assert dataset.has_wave_v is False
    assert dataset.wave5_uv is None


class TestSummaryValidation:
  """Malformed summaries are rejected at the loading boundary, not carried through."""

  @pytest.mark.parametrize(
    ("field", "value", "message"),
    [
      ("mean", float("nan"), "must be finite"),
      ("mean", float("inf"), "must be finite"),
      ("std", -0.1, "must not be negative"),
      ("n", 1.9, "must be an integer"),
      ("n", 0, "must be positive"),
    ],
    ids=["nan-mean", "inf-mean", "negative-sd", "fractional-n", "zero-n"],
  )
  def test_malformed_statistics_are_rejected(self, tmp_path, field, value, message):
    summary = _minimal_grouped_summary()
    summary["groups"]["nexp"]["wave1_uv"][field] = value
    _write_grouped_dataset(tmp_path, summary)
    with pytest.raises(ValueError, match=message) as error:
      load_abr_dataset("human", tmp_path)
    assert "nexp" in str(error.value)
    assert "wave1_uv" in str(error.value)

  @pytest.mark.parametrize("units", ["mV", "", None], ids=["millivolts", "empty", "missing"])
  def test_wave_amplitudes_must_declare_microvolts(self, tmp_path, units):
    summary = _minimal_grouped_summary()
    if units is None:
      del summary["measures"]["wave1_uv"]["units"]
    else:
      summary["measures"]["wave1_uv"]["units"] = units
    _write_grouped_dataset(tmp_path, summary)
    with pytest.raises(ValueError, match="wave amplitudes must be"):
      load_abr_dataset("human", tmp_path)

  def test_audiometry_keeps_its_own_units(self, tmp_path):
    # dB HL context must not be forced into the wave-amplitude unit set.
    _write_grouped_dataset(tmp_path, _minimal_grouped_summary())
    dataset = load_abr_dataset("human", tmp_path)
    assert dataset.context_measures["audiometric_lfa_db_hl"].units == "dB HL"

  def test_unknown_design_is_rejected(self, tmp_path, monkeypatch):
    _write_grouped_dataset(tmp_path, _minimal_grouped_summary())
    typo = dataclasses.replace(HUMAN_FILES, design="independant_groups")
    monkeypatch.setitem(SPECIES_DATA_FILES, "human", typo)
    with pytest.raises(ValueError, match="unsupported design"):
      load_abr_dataset("human", tmp_path)
