"""Tests for the human summary generator and the committed summary it produces."""

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


class TestCommittedSummaryIsUpToDate:
  """The committed JSON must be exactly what the script produces today."""

  def test_no_drift_between_script_and_committed_file(self, generator, tmp_path):
    regenerated = tmp_path / "human_abr_summary.json"
    generator.main(output_path=regenerated)
    assert json.loads(regenerated.read_text(encoding="utf-8")) == json.loads(
      COMMITTED_SUMMARY.read_text(encoding="utf-8")
    ), (
      "human_abr_summary.json is stale. Regenerate it with "
      "`uv run python scripts/generate_human_abr_summary.py`."
    )

  def test_main_writes_to_the_requested_path(self, generator, tmp_path):
    target = tmp_path / "nested" / "out.json"
    written = generator.main(output_path=target)
    assert written == target
    assert target.is_file()


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

  def test_group_sizes(self, summary):
    assert summary["groups"]["ctrl"]["n"] == 55
    assert summary["groups"]["nexp"]["n"] == 53
    assert summary["groups"]["ma"]["n"] == 58
    assert len(summary["subjects"]) == 166

  def test_wave_i_group_means(self, summary):
    assert summary["groups"]["ctrl"]["wave1_uv"]["mean"] == pytest.approx(1.1622, abs=1e-4)
    assert summary["groups"]["nexp"]["wave1_uv"]["mean"] == pytest.approx(1.0972, abs=1e-4)
    assert summary["groups"]["ma"]["wave1_uv"]["mean"] == pytest.approx(0.6137, abs=1e-4)

  def test_measures_with_missing_values_report_a_smaller_valid_n(self, summary):
    # A few subjects carry a literal "NaN" wave amplitude; they still count
    # towards the group n but not towards that measure's n.
    assert summary["groups"]["ctrl"]["wave1_uv"]["n"] == 54
    assert summary["groups"]["nexp"]["wave1_uv"]["n"] == 53
    assert summary["groups"]["ma"]["wave1_uv"]["n"] == 55

  def test_every_measure_declares_units(self, summary):
    for measure, stats in summary["measures"].items():
      assert stats["units"], f"{measure} is missing units"
    assert summary["measures"]["wave1_uv"]["units"] == "uV"
    assert summary["measures"]["audiometric_lfa_db_hl"]["units"] == "dB HL"

  def test_no_abr_thresholds_are_claimed(self, summary):
    # The dataset has no ABR thresholds. Behavioural dB HL audiometry must not
    # be presented as one, so no dB SPL threshold block may appear here.
    assert "thresholds_db_spl" not in summary
    assert all(not key.endswith("db_spl") for key in summary["groups"]["ctrl"])
