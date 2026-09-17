"""Tests for the empirical ABR dataset loader shared by both species."""

import json
import pathlib

import pytest

from carfac_ephys.empirical import (
  CLICK_FREQUENCY_HZ,
  DEFAULT_DATA_DIR,
  DEFAULT_SPECIES,
  SPECIES_DATA_FILES,
  AbrDataset,
  AnimalWaveAmplitudes,
  PrePostStat,
  load_abr_dataset,
)


@pytest.fixture(name="dataset", scope="module")
def dataset_fixture():
  """Loads the shipped chinchilla dataset once for all tests."""
  return load_abr_dataset("chinchilla")


def _write_dataset(
  directory: pathlib.Path,
  summary: dict,
  csv_rows: list[str],
  species: str = "chinchilla",
) -> pathlib.Path:
  """Writes a synthetic dataset for `species` into a directory and returns it.

  The per-subject CSV header uses `species`'s own `condition_column` (e.g.
  "TimePoint" for chinchilla, "Group" for human), so `csv_rows` must supply
  values from that species' `baseline_label`/`comparison_labels`.
  """
  files = SPECIES_DATA_FILES[species]
  (directory / files.summary_file_name).write_text(json.dumps(summary), encoding="utf-8")
  header = f"ID,{files.condition_column},W1,W5\n"
  (directory / files.per_subject_file_name).write_text(
    header + "\n".join(csv_rows) + "\n", encoding="utf-8"
  )
  return directory


def _minimal_summary(has_wave_v: bool = True) -> dict:
  """Returns a minimal well-formed summary structure with two frequencies.

  Args:
    has_wave_v: Whether to include a `high_level_w5_uv` block, matching how
      `SPECIES_DATA_FILES` marks which species report Wave-V.
  """
  waves = {
    "mean_pre": [2.0, 1.0, 4.0],
    "mean_post": [1.0, 0.5, 2.0],
    "std_pre": [0.2, 0.1, 0.4],
    "std_post": [0.1, 0.05, 0.2],
  }
  summary = {
    "source": "synthetic",
    "subjects": ["A1"],
    "frequencies_hz": [0, 4000],
    "thresholds_db_spl": {
      "mean_pre": [10.0, 20.0],
      "mean_post": [12.0, 25.0],
      "std_pre": [1.0, 2.0],
      "std_post": [1.5, 2.5],
    },
    "high_level_w1_uv": dict(waves),
  }
  if has_wave_v:
    summary["high_level_w5_uv"] = dict(waves)
  return summary


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
  """Tests for per-subject amplitude ratios."""

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


class TestSpeciesRegistry:
  """Tests for the species data file registry that replaces per-species classes."""

  def test_registered_species(self):
    assert "chinchilla" in SPECIES_DATA_FILES
    assert "human" in SPECIES_DATA_FILES
    assert SPECIES_DATA_FILES["chinchilla"].has_wave_v is True
    assert SPECIES_DATA_FILES["human"].has_wave_v is True

  def test_chinchilla_is_a_paired_timepoint_comparison(self):
    files = SPECIES_DATA_FILES["chinchilla"]
    assert files.paired is True
    assert files.condition_column == "TimePoint"
    assert files.baseline_label == "pre"
    assert files.comparison_labels == ("2wk",)

  def test_human_is_an_unpaired_group_comparison(self):
    files = SPECIES_DATA_FILES["human"]
    assert files.paired is False
    assert files.condition_column == "Group"
    assert files.baseline_label == "ctrl"
    assert files.comparison_labels == ("nexp", "ma")

  def test_default_species_is_registered(self):
    assert DEFAULT_SPECIES in SPECIES_DATA_FILES

  def test_unsupported_species_raises(self, tmp_path):
    with pytest.raises(ValueError, match="Unsupported species"):
      load_abr_dataset("axolotl", tmp_path)


class TestLoadShippedChinchillaDataset:
  """Tests against the chinchilla dataset shipped in the repository data directory."""

  def test_data_dir_exists(self):
    files = SPECIES_DATA_FILES["chinchilla"]
    assert (DEFAULT_DATA_DIR / files.summary_file_name).is_file()
    assert (DEFAULT_DATA_DIR / files.per_subject_file_name).is_file()

  def test_species_and_source(self, dataset):
    assert dataset.species == "chinchilla"
    assert "Bharadwaj" in dataset.source

  def test_subjects(self, dataset):
    assert len(dataset.subjects) == 7
    assert dataset.subjects[0] == "Q348"

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
    assert dataset.has_wave_v is True
    assert dataset.tone_average_w5_uv.ratio > dataset.tone_average_w1_uv.ratio

  def test_per_subject_matches_summary_subjects(self, dataset):
    assert tuple(subject.animal_id for subject in dataset.per_subject) == dataset.subjects

  def test_per_subject_ratios(self, dataset):
    ratios = dataset.per_animal_w1_ratios
    assert len(ratios) == 7
    assert all(ratio > 0.0 for ratio in ratios)

  def test_per_subject_values_match_csv(self, dataset):
    q348 = dataset.per_subject[0]
    assert q348.pre_w1_uv == pytest.approx(1.04276378865147)
    assert q348.post_w1_uv == pytest.approx(0.631910633176243)

  def test_unknown_frequency_raises(self, dataset):
    with pytest.raises(KeyError, match="available frequencies"):
      dataset.threshold_shift_db(12345.0)


class TestLoadSyntheticDataset:
  """Tests for loader behaviour on synthetic inputs, using chinchilla's
  paired (repeated-measures) TimePoint schema.

  Human's schema is unpaired (independent groups), not a same-subject
  pre/post pair, so it is exercised separately in
  `TestLoadSyntheticGroupedDataset` below rather than parametrized here.
  """

  @pytest.mark.parametrize("species,has_wave_v", [("chinchilla", True)])
  def test_loads_from_custom_directory(self, tmp_path, species, has_wave_v):
    _write_dataset(
      tmp_path,
      _minimal_summary(has_wave_v),
      ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"],
      species=species,
    )
    dataset = load_abr_dataset(species, tmp_path)
    assert isinstance(dataset, AbrDataset)
    assert dataset.species == species
    assert dataset.paired is True
    assert dataset.comparison_group == "2wk"
    assert dataset.source == "synthetic"
    assert dataset.tone_average_w1_uv.ratio == pytest.approx(0.5)
    assert dataset.per_animal_w1_ratios == pytest.approx((0.5,))
    assert dataset.has_wave_v is has_wave_v
    if not has_wave_v:
      assert dataset.high_level_w5_uv is None
      assert dataset.tone_average_w5_uv is None

  def test_accepts_string_path(self, tmp_path):
    _write_dataset(tmp_path, _minimal_summary(), ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    assert load_abr_dataset("chinchilla", str(tmp_path)).subjects == ("A1",)

  def test_missing_files_raise(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      load_abr_dataset("chinchilla", tmp_path)

  def test_mismatched_subjects_raise(self, tmp_path):
    _write_dataset(tmp_path, _minimal_summary(), ["B9,pre,2.0,4.0", "B9,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="Subject identifiers disagree"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_missing_time_point_raises(self, tmp_path):
    _write_dataset(tmp_path, _minimal_summary(), ["A1,pre,2.0,4.0"])
    with pytest.raises(ValueError, match="missing time points"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_missing_series_raises(self, tmp_path):
    summary = _minimal_summary()
    del summary["thresholds_db_spl"]["std_post"]
    _write_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="missing series"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_inconsistent_series_lengths_raise(self, tmp_path):
    summary = _minimal_summary()
    summary["thresholds_db_spl"]["std_post"] = [1.0]
    _write_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="differing lengths"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_unexpected_entry_count_raises(self, tmp_path):
    summary = _minimal_summary()
    for key in summary["high_level_w1_uv"]:
      summary["high_level_w1_uv"][key] = [1.0]
    _write_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="expected"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_missing_tone_average_raises(self, tmp_path):
    summary = _minimal_summary()
    for block in ("high_level_w1_uv", "high_level_w5_uv"):
      for key in summary[block]:
        summary[block][key] = summary[block][key][:2]
    _write_dataset(tmp_path, summary, ["A1,pre,2.0,4.0", "A1,2wk,1.0,2.0"])
    with pytest.raises(ValueError, match="trailing average entry"):
      load_abr_dataset("chinchilla", tmp_path)

  def test_empty_csv_raises(self, tmp_path):
    summary = _minimal_summary()
    files = SPECIES_DATA_FILES["chinchilla"]
    (tmp_path / files.summary_file_name).write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / files.per_subject_file_name).write_text("ID,TimePoint,W1,W5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No rows found"):
      load_abr_dataset("chinchilla", tmp_path)


def _minimal_grouped_summary() -> dict:
  """Returns a minimal well-formed summary for an unpaired (independent-groups)
  species with more than one `comparison_labels` option (human: "nexp"/"ma").

  Each statistics block is nested one level deeper, keyed by comparison
  label — see `_select_group_block` in `empirical.py` — since the baseline
  ("ctrl") side is shared but the comparison side differs by which group was
  requested; a single flat block (chinchilla's shape) can't represent both.
  The "nexp" and "ma" blocks deliberately use different `mean_post` values
  (aggregate post/pre ratio 0.5 vs 0.3) so a test can catch a `comparison_group`
  that silently keeps using one group's numbers regardless of the other.
  """
  return {
    "source": "synthetic",
    "subjects": ["A1", "A2", "B1", "B2", "C1"],
    "frequencies_hz": [0, 4000],
    "thresholds_db_spl": {
      "nexp": {
        "mean_pre": [10.0, 20.0],
        "mean_post": [12.0, 25.0],
        "std_pre": [1.0, 2.0],
        "std_post": [1.5, 2.5],
      },
      "ma": {
        "mean_pre": [10.0, 20.0],
        "mean_post": [15.0, 30.0],
        "std_pre": [1.0, 2.0],
        "std_post": [2.0, 3.0],
      },
    },
    "high_level_w1_uv": {
      "nexp": {
        "mean_pre": [5.0, 3.0, 2.0],
        "mean_post": [4.0, 2.0, 1.0],
        "std_pre": [0.5, 0.3, 0.2],
        "std_post": [0.4, 0.2, 0.1],
      },
      "ma": {
        "mean_pre": [5.0, 3.0, 2.0],
        "mean_post": [3.0, 1.5, 0.6],
        "std_pre": [0.5, 0.3, 0.2],
        "std_post": [0.3, 0.15, 0.06],
      },
    },
    "high_level_w5_uv": {
      "nexp": {
        "mean_pre": [8.0, 5.0, 4.0],
        "mean_post": [7.0, 4.0, 2.0],
        "std_pre": [0.8, 0.5, 0.4],
        "std_post": [0.7, 0.4, 0.2],
      },
      "ma": {
        "mean_pre": [8.0, 5.0, 4.0],
        "mean_post": [6.0, 3.5, 1.6],
        "std_pre": [0.8, 0.5, 0.4],
        "std_post": [0.6, 0.35, 0.16],
      },
    },
  }


# Two "ctrl" subjects, two "nexp" subjects, one "ma" subject — enough to
# exercise the default comparison group, an explicit alternate one, and rows
# from the *other* comparison group being excluded either way.
_GROUPED_ROWS = [
  "A1,ctrl,2.0,4.0",
  "A2,ctrl,2.2,4.2",
  "B1,nexp,1.0,2.0",
  "B2,nexp,1.2,2.2",
  "C1,ma,0.8,1.6",
]


class TestLoadSyntheticGroupedDataset:
  """Tests for the unpaired (independent-groups) loading path, e.g. human.

  Unlike `TestLoadSyntheticDataset` (chinchilla's paired pre/2wk design),
  here each subject appears under exactly one `Group` label, so there is no
  same-subject pairing and no per-subject ratio — only the group-level
  comparison is meaningful.
  """

  def test_default_comparison_group_is_first_registered_label(self, tmp_path):
    _write_dataset(tmp_path, _minimal_grouped_summary(), _GROUPED_ROWS, species="human")
    dataset = load_abr_dataset("human", tmp_path)
    assert dataset.comparison_group == "nexp"
    assert dataset.paired is False
    assert {s.animal_id for s in dataset.per_subject} == {"A1", "A2", "B1", "B2"}

  def test_explicit_comparison_group_selects_the_other_pair(self, tmp_path):
    _write_dataset(tmp_path, _minimal_grouped_summary(), _GROUPED_ROWS, species="human")
    dataset = load_abr_dataset("human", tmp_path, comparison_group="ma")
    assert dataset.comparison_group == "ma"
    assert {s.animal_id for s in dataset.per_subject} == {"A1", "A2", "C1"}

  def test_invalid_comparison_group_raises(self, tmp_path):
    _write_dataset(tmp_path, _minimal_grouped_summary(), _GROUPED_ROWS, species="human")
    with pytest.raises(ValueError, match="not valid for 'human'"):
      load_abr_dataset("human", tmp_path, comparison_group="bogus")

  def test_group_level_stats_actually_switch_with_comparison_group(self, tmp_path):
    # Regression test: mean_pre/mean_post must come from the *requested*
    # comparison label's nested block, not silently reuse whichever one
    # happens to be first. _minimal_grouped_summary gives "nexp" and "ma"
    # different post values specifically so this can't pass by accident.
    _write_dataset(tmp_path, _minimal_grouped_summary(), _GROUPED_ROWS, species="human")
    nexp_dataset = load_abr_dataset("human", tmp_path, comparison_group="nexp")
    ma_dataset = load_abr_dataset("human", tmp_path, comparison_group="ma")
    assert nexp_dataset.suprathreshold_w1_ratio == pytest.approx(0.5)
    assert ma_dataset.suprathreshold_w1_ratio == pytest.approx(0.3)
    assert nexp_dataset.click_threshold_shift_db != ma_dataset.click_threshold_shift_db

  def test_missing_nested_comparison_block_raises(self, tmp_path):
    summary = _minimal_grouped_summary()
    del summary["high_level_w1_uv"]["ma"]
    _write_dataset(tmp_path, summary, _GROUPED_ROWS, species="human")
    with pytest.raises(ValueError, match="no 'ma' entry"):
      load_abr_dataset("human", tmp_path, comparison_group="ma")

  def test_per_subject_has_only_its_own_group_populated(self, tmp_path):
    _write_dataset(tmp_path, _minimal_grouped_summary(), _GROUPED_ROWS, species="human")
    dataset = load_abr_dataset("human", tmp_path)
    by_id = {s.animal_id: s for s in dataset.per_subject}
    assert by_id["A1"].pre_w1_uv == pytest.approx(2.0)
    assert by_id["A1"].post_w1_uv is None
    assert by_id["B1"].pre_w1_uv is None
    assert by_id["B1"].post_w1_uv == pytest.approx(1.0)

  def test_per_subject_ratio_raises_for_unpaired_subject(self, tmp_path):
    _write_dataset(tmp_path, _minimal_grouped_summary(), _GROUPED_ROWS, species="human")
    dataset = load_abr_dataset("human", tmp_path)
    with pytest.raises(ValueError, match="unpaired design"):
      _ = dataset.per_subject[0].w1_ratio

  def test_per_animal_w1_ratios_is_empty_rather_than_raising(self, tmp_path):
    _write_dataset(tmp_path, _minimal_grouped_summary(), _GROUPED_ROWS, species="human")
    dataset = load_abr_dataset("human", tmp_path)
    assert dataset.per_animal_w1_ratios == ()

  def test_group_level_ratio_is_still_available(self, tmp_path):
    _write_dataset(tmp_path, _minimal_grouped_summary(), _GROUPED_ROWS, species="human")
    dataset = load_abr_dataset("human", tmp_path)
    assert dataset.suprathreshold_w1_ratio == pytest.approx(0.5)

  def test_case_insensitive_wave_columns(self, tmp_path):
    # The shipped Human_Synaptopathy_ABRdata.csv uses lowercase "w1"/"w5".
    files = SPECIES_DATA_FILES["human"]
    (tmp_path / files.summary_file_name).write_text(
      json.dumps(_minimal_grouped_summary()), encoding="utf-8"
    )
    (tmp_path / files.per_subject_file_name).write_text(
      "ID,Group,w1,w5\n" + "\n".join(_GROUPED_ROWS) + "\n", encoding="utf-8"
    )
    dataset = load_abr_dataset("human", tmp_path)
    assert len(dataset.per_subject) == 4  # ctrl (2) + nexp (2); "ma" excluded by default.

  def test_duplicate_subject_in_group_raises(self, tmp_path):
    rows = [*_GROUPED_ROWS, "A1,ctrl,9.9,9.9"]
    _write_dataset(tmp_path, _minimal_grouped_summary(), rows, species="human")
    with pytest.raises(ValueError, match="more than once"):
      load_abr_dataset("human", tmp_path)

  def test_unlisted_csv_subject_raises(self, tmp_path):
    rows = [*_GROUPED_ROWS, "Z9,ctrl,1.0,1.0"]  # "Z9" isn't in the summary's subjects.
    _write_dataset(tmp_path, _minimal_grouped_summary(), rows, species="human")
    with pytest.raises(ValueError, match="not listed in summary"):
      load_abr_dataset("human", tmp_path)