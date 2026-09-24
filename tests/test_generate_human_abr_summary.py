"""Tests for the human summary generator and the committed summary it produces."""

import csv
import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_human_abr_summary.py"
COMMITTED_SUMMARY = REPO_ROOT / "src" / "carfac_ephys" / "data" / "human_abr_summary.json"


def _load_generator():
  """Imports the generator script, which lives outside the installed package."""
  spec = importlib.util.spec_from_file_location("generate_human_abr_summary", SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.fixture(name="generator", scope="module")
def generator_fixture():
  return _load_generator()


@pytest.fixture(name="synthetic_csv")
def synthetic_csv_fixture(tmp_path):
  """Creates distinct groups so mixed-up rows change the expected statistics."""
  observations = {
    "ctrl": [(1, 2), (3, 4), ("NaN", 6)],
    "nexp": [(4, 10), (8, 14)],
    "ma": [(6, 20), ("NaN", 24), (10, "NaN"), (14, 28)],
  }
  audiometry = {"ctrl": (10, 20, 30), "nexp": (11, 21, 31), "ma": (12, 22, 32)}
  path = tmp_path / "synthetic.csv"
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["ID", "Group", "w1", "w5", "LFA", "HFA", "EHFA"])
    for group, waves in observations.items():
      for index, (wave1, wave5) in enumerate(waves):
        writer.writerow([f"invented-{group}-{index}", group, wave1, wave5, *audiometry[group]])
  return path


def _assert_aggregate_only(summary):
  """Allows only group counts and aggregate measure statistics in the output."""
  assert set(summary) == {
    "source",
    "design",
    "baseline_group",
    "comparison_groups",
    "measures",
    "groups",
    "notes",
  }
  for group in summary["groups"].values():
    assert set(group) == {"n", *summary["measures"]}
    for measure in summary["measures"]:
      assert set(group[measure]) == {"mean", "std", "n"}


class TestSyntheticSource:
  @pytest.mark.parametrize(
    ("group_name", "group_n", "wave1", "wave5", "audiometry"),
    [
      ("ctrl", 3, (2.0, 2**0.5, 2), (4.0, 2.0, 3), (10.0, 20.0, 30.0)),
      ("nexp", 2, (6.0, 8**0.5, 2), (12.0, 8**0.5, 2), (11.0, 21.0, 31.0)),
      ("ma", 4, (10.0, 4.0, 3), (24.0, 4.0, 3), (12.0, 22.0, 32.0)),
    ],
    ids=["ctrl", "nexp", "ma"],
  )
  def test_group_statistics_and_missing_values(
    self, generator, synthetic_csv, group_name, group_n, wave1, wave5, audiometry
  ):
    summary = generator.build_summary(synthetic_csv)

    group = summary["groups"][group_name]
    assert group["n"] == group_n
    # Expected means and sample SDs are calculated independently of the generator.
    for measure, (mean, std, valid_n) in zip(("wave1_uv", "wave5_uv"), (wave1, wave5)):
      assert group[measure] == {
        "mean": pytest.approx(mean),
        "std": pytest.approx(std),
        "n": valid_n,
      }
    for band, mean in zip(("lfa", "hfa", "ehfa"), audiometry):
      assert group[f"audiometric_{band}_db_hl"] == {"mean": mean, "std": 0.0, "n": group_n}

  def test_identifiers_are_not_emitted(self, generator, synthetic_csv):
    summary = generator.build_summary(synthetic_csv)

    _assert_aggregate_only(summary)
    assert "invented-" not in json.dumps(summary)

  def test_identifier_column_is_not_required(self, generator, synthetic_csv):
    expected = generator.build_summary(synthetic_csv)
    with synthetic_csv.open(newline="", encoding="utf-8") as handle:
      rows = list(csv.reader(handle))
    with synthetic_csv.open("w", newline="", encoding="utf-8") as handle:
      csv.writer(handle).writerows(row[1:] for row in rows)

    assert generator.build_summary(synthetic_csv) == expected


class TestSourceValidation:
  """Malformed source rows must fail generation, not slip into the summary."""

  HEADER = "ID,Group,w1,w5,LFA,HFA,EHFA\n"
  VALID_ROWS = (
    "invented-1,ctrl,1,2,10,20,30\n"
    "invented-2,ctrl,3,4,10,20,30\n"
    "invented-3,nexp,5,6,11,21,31\n"
    "invented-4,nexp,7,8,11,21,31\n"
    "invented-5,ma,9,10,12,22,32\n"
    "invented-6,ma,11,12,12,22,32\n"
  )

  def _write(self, tmp_path, body):
    path = tmp_path / "source.csv"
    path.write_text(self.HEADER + body, encoding="utf-8")
    return path

  def test_unexpected_group_is_rejected(self, generator, tmp_path):
    body = self.VALID_ROWS + "invented-7,typo,1,2,10,20,30\n"
    with pytest.raises(ValueError, match="unexpected group 'typo'"):
      generator.build_summary(self._write(tmp_path, body))

  def test_infinite_observation_is_rejected(self, generator, tmp_path):
    body = self.VALID_ROWS.replace("invented-1,ctrl,1,2", "invented-1,ctrl,inf,2")
    with pytest.raises(ValueError, match="infinite observation"):
      generator.build_summary(self._write(tmp_path, body))

  def test_single_valid_observation_is_rejected(self, generator, tmp_path):
    body = self.VALID_ROWS.replace("invented-2,ctrl,3,4", "invented-2,ctrl,NaN,4")
    with pytest.raises(ValueError, match="at least 2 are needed"):
      generator.build_summary(self._write(tmp_path, body))


class TestCommittedSummaryIsUpToDate:
  """The committed JSON must be exactly what the script produces today."""

  def test_no_drift_between_script_and_committed_file(self, generator, tmp_path):
    if not generator.DEFAULT_INPUT_CSV.is_file():
      pytest.skip("Private human ABR source CSV is unavailable.")
    regenerated = tmp_path / "human_abr_summary.json"
    generator.main(output_path=regenerated)
    assert json.loads(regenerated.read_text(encoding="utf-8")) == json.loads(
      COMMITTED_SUMMARY.read_text(encoding="utf-8")
    ), (
      "human_abr_summary.json is stale. Regenerate it with "
      "`uv run python scripts/generate_human_abr_summary.py`."
    )

  def test_main_writes_to_the_requested_path(self, generator, synthetic_csv, tmp_path):
    target = tmp_path / "nested" / "out.json"
    written = generator.main(output_path=target, csv_path=synthetic_csv)
    assert written == target
    assert json.loads(target.read_text(encoding="utf-8")) == generator.build_summary(synthetic_csv)


@pytest.fixture(name="summary", scope="module")
def summary_fixture():
  """Reads the committed summary once for all shape tests."""
  return json.loads(COMMITTED_SUMMARY.read_text(encoding="utf-8"))


class TestSummaryShape:
  """Pins the schema and the shipped values, so a silent data change fails here."""

  def test_declares_an_independent_groups_design(self, summary):
    assert summary["design"] == "independent_groups"
    assert summary["baseline_group"] == "ctrl"
    assert summary["comparison_groups"] == ["nexp", "ma"]

  def test_contains_only_aggregate_data(self, summary):
    _assert_aggregate_only(summary)

  def test_group_sizes(self, summary):
    assert summary["groups"]["ctrl"]["n"] == 55
    assert summary["groups"]["nexp"]["n"] == 53
    assert summary["groups"]["ma"]["n"] == 58
    assert sum(group["n"] for group in summary["groups"].values()) == 166

  def test_wave_i_group_means(self, summary):
    assert summary["groups"]["ctrl"]["wave1_uv"]["mean"] == pytest.approx(1.1622, abs=1e-4)
    assert summary["groups"]["nexp"]["wave1_uv"]["mean"] == pytest.approx(1.0972, abs=1e-4)
    assert summary["groups"]["ma"]["wave1_uv"]["mean"] == pytest.approx(0.6137, abs=1e-4)

  def test_measures_with_missing_values_report_a_smaller_valid_n(self, summary):
    # Missing amplitudes reduce the measure count, but not the group count.
    assert summary["groups"]["ctrl"]["wave1_uv"]["n"] == 54
    assert summary["groups"]["nexp"]["wave1_uv"]["n"] == 53
    assert summary["groups"]["ma"]["wave1_uv"]["n"] == 55

  def test_every_measure_declares_units(self, summary):
    for measure, stats in summary["measures"].items():
      assert stats["units"], f"{measure} is missing units"
    assert summary["measures"]["wave1_uv"]["units"] == "uV"
    assert summary["measures"]["audiometric_lfa_db_hl"]["units"] == "dB HL"

  def test_no_abr_thresholds_are_claimed(self, summary):
    # Behavioural audiometry must not be presented as an ABR threshold.
    assert "thresholds_db_spl" not in summary
    assert all(not key.endswith("db_spl") for key in summary["groups"]["ctrl"])
