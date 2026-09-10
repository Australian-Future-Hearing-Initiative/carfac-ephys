"""Tests for cohort definitions, level series simulations, and Click CLI."""

import pathlib
from typing import Any

from click.testing import CliRunner
import pytest

from carfac_ephys.cli import main
from carfac_ephys.experiment import (
  DEFAULT_COHORT_CONDITIONS,
  Cohort,
  CohortCondition,
  format_ascii_table,
  get_default_cohort,
  plot_abr_growth,
  plot_efr_growth,
  simulate_abr_level_series,
  simulate_efr_level_series,
)


class TestCohortDefinitions:
  """Tests for CohortCondition and Cohort collection initialization."""

  def test_default_cohort_conditions(self):
    # Verify default 5 conditions and biophysical values.
    cohort = Cohort()
    assert len(cohort) == 5
    assert "Control" in cohort
    assert "Synaptopathy-50" in cohort
    assert "Synaptopathy-25" in cohort
    assert "OHC-Loss" in cohort
    assert "Mixed-Loss" in cohort

    # Validate biophysical impairment parameters.
    assert cohort["Control"].ohc_health == 1.0
    assert cohort["Control"].fiber_retention == 1.0

    assert cohort["Synaptopathy-50"].ohc_health == 1.0
    assert cohort["Synaptopathy-50"].fiber_retention == 0.5

    assert cohort["Synaptopathy-25"].ohc_health == 1.0
    assert cohort["Synaptopathy-25"].fiber_retention == 0.25

    assert cohort["OHC-Loss"].ohc_health == 0.4
    assert cohort["OHC-Loss"].fiber_retention == 1.0

    assert cohort["Mixed-Loss"].ohc_health == 0.4
    assert cohort["Mixed-Loss"].fiber_retention == 0.5

  def test_cohort_aliasing(self):
    # Support lookup alias for Control condition.
    cohort = Cohort()
    assert cohort["Control (Healthy)"] == cohort["Control"]
    assert "Control (Healthy)" in cohort

  def test_get_default_cohort_helper(self):
    # Verify helper function returns complete Cohort instance.
    cohort = get_default_cohort()
    assert isinstance(cohort, Cohort)
    assert len(cohort.conditions) == len(DEFAULT_COHORT_CONDITIONS)

  def test_custom_cohort_initialization(self):
    # Initialize from dictionary of tuples.
    custom_dict: dict[str, Any] = {
      "Mild": (0.8, 0.7),
      "Severe": (0.2, 0.1),
    }
    cohort = Cohort(custom_dict)
    assert len(cohort) == 2
    assert cohort["Mild"].ohc_health == 0.8
    assert cohort["Mild"].fiber_retention == 0.7
    assert cohort["Severe"].ohc_health == 0.2
    assert cohort["Severe"].fiber_retention == 0.1

    # Initialize from sequence of CohortCondition instances.
    cond_list = [
      CohortCondition(name="Custom-A", ohc_health=0.9, fiber_retention=0.8),
      CohortCondition(name="Custom-B", ohc_health=0.5, fiber_retention=0.3),
    ]
    cohort_seq = Cohort(cond_list)
    assert len(cohort_seq) == 2
    assert cohort_seq["Custom-A"].name == "Custom-A"

  def test_invalid_cohort_inputs(self):
    with pytest.raises(TypeError):
      Cohort(12345)  # type: ignore

    with pytest.raises(ValueError, match="length 2 or 3"):
      Cohort({"Invalid": (1.0, 1.0, "desc", "extra")})  # type: ignore


class TestSimulateLevelSeries:
  """Tests for simulate_abr_level_series and simulate_efr_level_series."""

  def test_simulate_abr_level_series_mini_sweep(self):
    # Run mini-sweep with 2 levels across Control and Synaptopathy-50.
    mini_cohort = Cohort(
      {
        "Control": (1.0, 1.0),
        "Synaptopathy-50": (1.0, 0.5),
        "OHC-Loss": (0.4, 1.0),
      }
    )
    click_levels = [60.0, 80.0]

    results = simulate_abr_level_series(
      cohort=mini_cohort,
      click_levels_db=click_levels,
      duration_s=0.02,
    )

    assert set(results.keys()) == {"Control", "Synaptopathy-50", "OHC-Loss"}
    assert len(results["Control"]) == 2
    assert len(results["Synaptopathy-50"]) == 2

    # Verify monotonic growth with level.
    assert results["Control"][1] > results["Control"][0]
    assert results["Synaptopathy-50"][1] > results["Synaptopathy-50"][0]

    # Verify synaptopathy reduces suprathreshold amplitude.
    assert results["Synaptopathy-50"][1] < results["Control"][1]

    # Verify OHC-Loss produces drastically smaller amplitude than Control at 60 dB.
    assert results["OHC-Loss"][0] < 0.1 * results["Control"][0]

  def test_simulate_abr_parameter_validation(self):
    with pytest.raises(ValueError, match="sample_rate"):
      simulate_abr_level_series(sample_rate=-1)

    with pytest.raises(ValueError, match="empty"):
      simulate_abr_level_series(click_levels_db=[])

  def test_simulate_efr_level_series_mini_sweep(self):
    # Run mini-sweep with 2 levels across Control and Synaptopathy-50.
    mini_cohort = Cohort(
      {
        "Control": (1.0, 1.0),
        "Synaptopathy-50": (1.0, 0.5),
      }
    )
    efr_levels = [60.0, 80.0]

    results = simulate_efr_level_series(
      cohort=mini_cohort,
      efr_levels_db=efr_levels,
      fc_hz=2000.0,
      fm_hz=100.0,
      duration_s=0.1,
    )

    assert set(results.keys()) == {"Control", "Synaptopathy-50"}
    assert len(results["Control"]) == 2
    assert len(results["Synaptopathy-50"]) == 2

    # Verify monotonic growth with level.
    assert results["Control"][1] > results["Control"][0]

    # Verify synaptopathy reduces suprathreshold EFR.
    assert results["Synaptopathy-50"][1] < results["Control"][1]

  def test_simulate_efr_parameter_validation(self):
    with pytest.raises(ValueError, match="sample_rate"):
      simulate_efr_level_series(sample_rate=0)

    with pytest.raises(ValueError, match="positive"):
      simulate_efr_level_series(fc_hz=-100.0)

    with pytest.raises(ValueError, match="Nyquist"):
      simulate_efr_level_series(sample_rate=1000, fm_hz=600.0)

    with pytest.raises(ValueError, match="empty"):
      simulate_efr_level_series(efr_levels_db=[])


class TestAsciiTableFormatter:
  """Tests for format_ascii_table."""

  def test_table_output_formatting(self):
    levels = [60.0, 80.0]
    results = {
      "Control": [10.5, 65.2],
      "Synaptopathy-50": [5.8, 33.1],
    }
    table = format_ascii_table("ABR Wave-I Growth", levels, results)

    assert "ABR Wave-I Growth" in table
    assert "Condition" in table
    assert "60 dB SPL" in table
    assert "80 dB SPL" in table
    assert "Control" in table
    assert "Synaptopathy-50" in table
    assert "10.5000" in table
    assert "65.2000" in table


class TestPlottingFunctions:
  """Tests for plot_abr_growth and plot_efr_growth."""

  def test_plot_abr_growth(self, tmp_path: pathlib.Path):
    levels = [60.0, 80.0]
    results = {
      "Control": [10.5, 65.2],
      "Synaptopathy-50": [5.8, 33.1],
    }
    fig_path = tmp_path / "abr_wave_i_growth.png"
    out_path = plot_abr_growth(levels, results, fig_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0

  def test_plot_efr_growth(self, tmp_path: pathlib.Path):
    levels = [60.0, 80.0]
    results = {
      "Control": [2.4, 5.8],
      "Synaptopathy-50": [1.4, 2.9],
    }
    fig_path = tmp_path / "efr_growth.png"
    out_path = plot_efr_growth(levels, results, fig_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


class TestCli:
  """Tests for Click CLI runner."""

  def test_cli_help(self):
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "carfac-ephys-simulate" in result.output or "output-dir" in result.output
    assert "--output-dir" in result.output
    assert "--plot" in result.output

  def test_cli_quick_no_plot(self):
    runner = CliRunner()
    result = runner.invoke(main, ["--quick", "--no-plot"])
    assert result.exit_code == 0
    assert "ABR Wave-I" in result.output
    assert "EFR Spectral Magnitude" in result.output
    assert "Saved" not in result.output

  def test_cli_quick_with_plot(self, tmp_path: pathlib.Path):
    runner = CliRunner()
    result = runner.invoke(
      main,
      [
        "--quick",
        "--plot",
        "--output-dir",
        str(tmp_path),
      ],
    )
    assert result.exit_code == 0
    assert "Saved ABR Wave-I growth figure" in result.output
    assert "Saved EFR growth figure" in result.output
    assert (tmp_path / "abr_wave_i_growth.png").exists()
    assert (tmp_path / "efr_growth.png").exists()
    assert (tmp_path / "abr_wave_i_growth.png").stat().st_size > 0
    assert (tmp_path / "efr_growth.png").stat().st_size > 0
