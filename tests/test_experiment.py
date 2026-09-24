"""Tests for cohort definitions, level series simulations, and Click CLI."""

import pathlib
from typing import Any

import pytest
from click.testing import CliRunner

from carfac_ephys.carfac_model import FiberRetention
from carfac_ephys.cli import main
from carfac_ephys.empirical import load_abr_dataset
from carfac_ephys.experiment import (
  DEFAULT_COHORT_CONDITIONS,
  THRESHOLD_SHIFT_TOLERANCE_DB,
  W1_RATIO_TOLERANCE,
  BiologicalValidation,
  Cohort,
  CohortCondition,
  compare_to_empirical,
  estimate_threshold_db,
  fit_response_scale_uv_per_au,
  format_ascii_table,
  format_empirical_comparison_table,
  format_markdown_table,
  generate_simulation_report,
  get_default_cohort,
  plot_abr_growth,
  plot_efr_growth,
  plot_empirical_comparison,
  simulate_abr_level_series,
  simulate_efr_level_series,
  validate_biological_signatures,
)


class TestCohortDefinitions:
  """Tests for CohortCondition and Cohort collection initialization."""

  def test_default_cohort_conditions(self):
    # Verify default 6 conditions and biophysical values.
    cohort = Cohort()
    assert len(cohort) == 6
    assert "Control" in cohort
    assert "Synaptopathy-50" in cohort
    assert "Synaptopathy-25" in cohort
    assert "Selective-Synaptopathy" in cohort
    assert "OHC-Loss" in cohort
    assert "Mixed-Loss" in cohort

    # Validate biophysical impairment parameters.
    assert cohort["Control"].ohc_health == 1.0
    assert cohort["Control"].fiber_retention == 1.0

    assert cohort["Synaptopathy-50"].ohc_health == 1.0
    assert cohort["Synaptopathy-50"].fiber_retention == 0.5

    assert cohort["Synaptopathy-25"].ohc_health == 1.0
    assert cohort["Synaptopathy-25"].fiber_retention == 0.25

    assert cohort["Selective-Synaptopathy"].ohc_health == 1.0
    assert cohort["Selective-Synaptopathy"].fiber_retention == FiberRetention(
      hsr=1.0, msr=0.5, lsr=0.0
    )

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

  def test_zero_modulation_depth_has_no_efr(self):
    # An unmodulated carrier must not produce an envelope response; the offset
    # ramp used to leak into the analysis window and create a spurious floor.
    mini_cohort = Cohort({"Control": (1.0, 1.0)})
    unmodulated = simulate_efr_level_series(cohort=mini_cohort, efr_levels_db=[80.0], depth=0.0)
    modulated = simulate_efr_level_series(cohort=mini_cohort, efr_levels_db=[80.0], depth=1.0)

    assert unmodulated["Control"][0] < 0.01 * modulated["Control"][0]

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
    # The calibration line is shared with the markdown report; the terminal
    # echoes it verbatim, so it must not carry markdown markup.
    assert "Fitted scale factor" in result.output
    assert "**" not in result.output
    # The quick sweep cannot bracket the threshold criterion, so that row is
    # skipped while the 80 dB SPL ratio is still reported.
    assert "Dataset: chinchilla (2wk vs pre)" in result.output
    assert "Validation against the chinchilla dataset" in result.output
    assert "| Click ABR threshold shift (dB) | n/a |" in result.output
    assert "Suprathreshold Wave-I ratio" in result.output

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
    assert "Saved empirical comparison figure" in result.output
    assert "Saved simulation report" in result.output
    assert (tmp_path / "abr_wave_i_growth.png").exists()
    assert (tmp_path / "efr_growth.png").exists()
    assert (tmp_path / "empirical_comparison_chinchilla.png").exists()
    assert (tmp_path / "simulation_report_chinchilla.md").exists()
    assert (tmp_path / "abr_wave_i_growth.png").stat().st_size > 0
    assert (tmp_path / "efr_growth.png").stat().st_size > 0
    assert (tmp_path / "empirical_comparison_chinchilla.png").stat().st_size > 0
    assert (tmp_path / "simulation_report_chinchilla.md").stat().st_size > 0

    # The quick sweep omits the 30-50 dB SPL levels, so the threshold criteria
    # have no data and must not be reported as failures.
    report = (tmp_path / "simulation_report_chinchilla.md").read_text(encoding="utf-8")
    assert "OHC Loss Threshold Shift (30-50 dB SPL)**: SKIPPED" in report
    assert "FAILED" not in report


class TestMarkdownTableFormatter:
  """Tests for format_markdown_table."""

  def test_markdown_table_formatting(self):
    levels = [60.0, 80.0]
    results = {
      "Control": [10.5, 65.2],
      "Synaptopathy-50": [5.8, 33.1],
    }
    table = format_markdown_table(levels, results)

    assert "| Condition | 60 dB SPL | 80 dB SPL |" in table
    assert "| --- | --- | --- |" in table
    assert "| Control | 10.5000 | 65.2000 |" in table
    assert "| Synaptopathy-50 | 5.8000 | 33.1000 |" in table


class TestBiologicalValidation:
  """Tests for validate_biological_signatures."""

  def test_validation_passes_on_simulation_results(self):
    click_levels = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    abr_results = {
      "Control": [0.0128, 0.1282, 1.3399, 10.7559, 51.5883, 67.5298],
      "Synaptopathy-50": [0.0075, 0.0760, 0.7949, 6.1125, 28.1907, 35.5266],
      "Synaptopathy-25": [0.0041, 0.0414, 0.4330, 3.2536, 14.6364, 18.2008],
      "Selective-Synaptopathy": [0.0123, 0.1238, 1.2915, 10.0933, 35.3127, 46.6437],
      "OHC-Loss": [0.0008, 0.0025, 0.0078, 0.0256, 0.1077, 0.8301],
      "Mixed-Loss": [0.0004, 0.0012, 0.0039, 0.0128, 0.0547, 0.4261],
    }
    efr_levels = [40.0, 50.0, 60.0, 70.0, 80.0]
    efr_results = {
      "Control": [1.0298, 1.7108, 2.4846, 3.5882, 5.8168],
      "Synaptopathy-50": [0.6961, 0.9871, 1.4827, 2.0362, 2.8883],
      "Synaptopathy-25": [0.3820, 0.6320, 0.6988, 1.1799, 1.9063],
      "Selective-Synaptopathy": [0.9641, 1.3883, 1.8917, 2.9321, 4.6401],
      "OHC-Loss": [0.0036, 0.0362, 0.3421, 1.7567, 4.7608],
      "Mixed-Loss": [0.0022, 0.0219, 0.2073, 1.0152, 2.5421],
    }

    val = validate_biological_signatures(
      click_levels_db=click_levels,
      abr_results=abr_results,
      efr_levels_db=efr_levels,
      efr_results=efr_results,
    )

    assert isinstance(val, BiologicalValidation)
    assert val.synaptopathy_low_preserved is True
    assert val.synaptopathy_high_scaled_50 is True
    assert val.synaptopathy_high_scaled_25 is True
    assert val.selective_low_level_spared is True
    assert val.empirical_threshold_shift_matched is True
    assert val.empirical_w1_ratio_matched is True
    assert val.efr_suprathreshold_drop is True
    assert val.ohc_threshold_shifted is True
    assert val.ohc_compression_lost is True
    assert val.mixed_loss_dual_deficit is True
    assert val.all_passed is True

  def test_selective_synaptopathy_criteria_fail_independently(self):
    click_levels = [40.0, 80.0]
    efr_levels, efr_results = [80.0], {"Control": [5.8]}

    # Sparing HSR fibers only matters if the low-level response survives; a
    # selective cohort that collapses near threshold is not selective.
    collapsed_low = {"Control": [0.1282, 67.5298], "Selective-Synaptopathy": [0.0500, 46.6437]}
    val = validate_biological_signatures(click_levels, collapsed_low, efr_levels, efr_results)

    assert val.selective_low_level_spared is False
    assert val.empirical_w1_ratio_matched is True
    assert val.all_passed is False

    # Matching Control at 80 dB SPL means losing the LSR and half the MSR fibers
    # cost nothing suprathreshold, which the animal ratio of ~0.74 rules out.
    no_attenuation = {"Control": [0.1282, 67.5298], "Selective-Synaptopathy": [0.1238, 66.0000]}
    val = validate_biological_signatures(click_levels, no_attenuation, efr_levels, efr_results)

    assert val.selective_low_level_spared is True
    assert val.empirical_w1_ratio_matched is False
    assert val.all_passed is False

  def test_validation_fails_on_unpreserved_synaptopathy(self):
    click_levels = [40.0, 80.0]
    abr_results = {
      "Control": [0.1, 60.0],
      "Synaptopathy-50": [0.01, 30.0],  # 0.01 < 0.3 * 0.1 -> fails
    }
    efr_levels = [80.0]
    efr_results = {
      "Control": [5.0],
      "Synaptopathy-50": [2.5],
    }
    val = validate_biological_signatures(click_levels, abr_results, efr_levels, efr_results)
    assert val.synaptopathy_low_preserved is False
    assert val.all_passed is False

  def test_missing_levels_are_skipped_not_failed(self):
    # Quick sweep omits the 30-50 dB SPL levels the threshold criteria need.
    click_levels = [60.0, 80.0]
    abr_results = {
      "Control": [10.7559, 67.5298],
      "Synaptopathy-50": [6.1125, 35.5266],
      "Synaptopathy-25": [3.2536, 18.2008],
      "OHC-Loss": [0.0256, 0.8301],
      "Mixed-Loss": [0.0128, 0.4261],
    }
    efr_levels = [60.0, 80.0]
    efr_results = {
      "Control": [2.4846, 5.8168],
      "Synaptopathy-50": [1.4827, 2.8883],
      "OHC-Loss": [0.3421, 4.7608],
      "Mixed-Loss": [0.2073, 2.5421],
    }

    val = validate_biological_signatures(click_levels, abr_results, efr_levels, efr_results)

    assert val.synaptopathy_low_preserved is None
    assert val.ohc_threshold_shifted is None
    assert val.selective_low_level_spared is None
    assert val.synaptopathy_high_scaled_50 is True
    assert val.any_skipped is True
    assert val.all_passed is True

  def test_report_marks_skipped_checks(self):
    content = generate_simulation_report(
      click_levels_db=[60.0, 80.0],
      abr_results={"Control": [10.0, 67.5298], "Synaptopathy-50": [5.0, 35.5]},
      efr_levels_db=[60.0, 80.0],
      efr_results={"Control": [2.5, 5.8], "Synaptopathy-50": [1.5, 2.9]},
    )

    assert "SKIPPED" in content
    assert "ALL CHECKS PASSED" not in content
    assert "SOME CHECKS SKIPPED" in content

  def test_report_failure_outranks_skipped_sibling(self):
    # Synaptopathy-50 is far below the expected ratio and Synaptopathy-25 was not
    # simulated; the combined criterion must report the failure, not the gap.
    content = generate_simulation_report(
      click_levels_db=[60.0, 80.0],
      abr_results={"Control": [10.0, 67.5298], "Synaptopathy-50": [5.0, 10.0]},
      efr_levels_db=[60.0, 80.0],
      efr_results={"Control": [2.5, 5.8], "Synaptopathy-50": [1.5, 2.9]},
    )

    assert "Synaptopathy Suprathreshold Scaling (80 dB SPL)**: FAILED" in content
    assert "SOME CHECKS FAILED" in content

  def test_cli_human_dataset_reports_exploratory_and_names_files_by_group(self, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
      main,
      ["--quick", "--species", "human", "--comparison-group", "ma", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "Dataset: human (ma vs ctrl)" in result.output
    assert "Exploratory comparison against the human dataset" in result.output
    assert "PASSED" not in result.output
    # The file the report names must be the file the CLI writes.
    assert (tmp_path / "empirical_comparison_human_ma.png").exists()
    report_path = tmp_path / "simulation_report_human_ma.md"
    assert report_path.exists()
    assert "empirical_comparison_human_ma.png" in report_path.read_text(encoding="utf-8")


class TestGenerateSimulationReport:
  """Tests for generate_simulation_report."""

  def test_generate_report_content_and_file(self, tmp_path: pathlib.Path):
    click_levels = [40.0, 80.0]
    abr_results = {
      "Control": [0.1282, 67.5298],
      "Synaptopathy-50": [0.0760, 35.5266],
      "Synaptopathy-25": [0.0414, 18.2008],
      "Selective-Synaptopathy": [0.1238, 46.6437],
      "OHC-Loss": [0.0025, 0.8301],
      "Mixed-Loss": [0.0012, 0.4261],
    }
    efr_levels = [40.0, 60.0, 80.0]
    efr_results = {
      "Control": [1.0298, 2.4846, 5.8168],
      "Synaptopathy-50": [0.6961, 1.4827, 2.8883],
      "Synaptopathy-25": [0.3820, 0.6988, 1.9063],
      "Selective-Synaptopathy": [0.9641, 1.8917, 4.6401],
      "OHC-Loss": [0.0036, 0.3421, 4.7608],
      "Mixed-Loss": [0.0022, 0.2073, 2.5421],
    }

    report_path = tmp_path / "report.md"
    content = generate_simulation_report(
      click_levels_db=click_levels,
      abr_results=abr_results,
      efr_levels_db=efr_levels,
      efr_results=efr_results,
      output_path=report_path,
    )

    assert "# CARFAC Electrophysiology Cohort Simulation Report" in content
    assert "Bharadwaj" in content
    assert "Synaptopathy-50" in content
    assert "Suprathreshold Wave-I Attenuation**: PASSED" in content
    assert "## 5. Empirical Comparison with Animal Data" in content
    assert "Measured (chinchilla (2wk))" in content
    assert "empirical_comparison_chinchilla.png" in content
    assert "ABR Wave-I Onset Amplitude (AU)" in content
    assert "spikes/s" not in content
    assert "Fitted scale factor" in content
    assert "EFR Spectral Magnitude at 100 Hz (AU)" in content
    assert "ALL CHECKS PASSED" in content

    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8") == content


class TestFitResponseScale:
  """Tests for fit_response_scale_uv_per_au."""

  def test_scale_maps_control_response_to_empirical_amplitude(self):
    click_levels = [60.0, 80.0]
    abr_results = {"Control": [10.0, 67.5298]}
    scale = fit_response_scale_uv_per_au(click_levels, abr_results)

    dataset = load_abr_dataset()
    expected_uv = dataset.baseline_reference_w1_uv
    assert scale * 67.5298 == pytest.approx(expected_uv)
    assert scale > 0.0

  def test_missing_calibration_level_is_rejected(self):
    with pytest.raises(ValueError, match="calibrate"):
      fit_response_scale_uv_per_au([60.0], {"Control": [10.0]})

  def test_report_notes_unavailable_scale_without_calibration_level(self):
    content = generate_simulation_report(
      click_levels_db=[60.0],
      abr_results={"Control": [10.0]},
      efr_levels_db=[60.0],
      efr_results={"Control": [1.0]},
    )
    assert "Scale factor unavailable" in content


class TestEstimateThresholdDb:
  """Tests for estimate_threshold_db."""

  def test_interpolates_crossing_on_log_amplitude_axis(self):
    # A decade of growth over 10 dB puts the half-decade criterion mid-segment.
    threshold = estimate_threshold_db([50.0, 60.0], [1.0, 10.0], criterion=10.0**0.5)
    assert threshold == pytest.approx(55.0)

  def test_unordered_levels_give_the_same_threshold(self):
    threshold = estimate_threshold_db([60.0, 50.0], [10.0, 1.0], criterion=10.0**0.5)
    assert threshold == pytest.approx(55.0)

  def test_criterion_outside_the_sweep_is_unresolved(self):
    # Above every simulated response, and below the lowest one.
    assert estimate_threshold_db([50.0, 60.0], [1.0, 10.0], criterion=100.0) is None
    assert estimate_threshold_db([50.0, 60.0], [1.0, 10.0], criterion=0.5) is None

  def test_zero_baseline_falls_back_to_linear_interpolation(self):
    threshold = estimate_threshold_db([50.0, 60.0], [0.0, 2.0], criterion=1.0)
    assert threshold == pytest.approx(55.0)

  def test_invalid_inputs_are_rejected(self):
    with pytest.raises(ValueError, match="criterion"):
      estimate_threshold_db([50.0, 60.0], [1.0, 10.0], criterion=0.0)
    with pytest.raises(ValueError, match="equally long"):
      estimate_threshold_db([50.0, 60.0], [1.0], criterion=1.0)


# Full-sweep simulation results, used as the reference for empirical comparison.
FULL_SWEEP_CLICK_LEVELS = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
FULL_SWEEP_ABR_RESULTS = {
  "Control": [0.0128, 0.1282, 1.3399, 10.7559, 51.5883, 67.5298],
  "Selective-Synaptopathy": [0.0123, 0.1238, 1.2915, 10.0933, 35.3127, 46.6437],
}


class TestCompareToEmpirical:
  """Tests for compare_to_empirical."""

  def test_selective_cohort_matches_the_reference_dataset(self):
    dataset = load_abr_dataset()
    comparison = compare_to_empirical(FULL_SWEEP_CLICK_LEVELS, FULL_SWEEP_ABR_RESULTS)

    # Both reference values come straight from the chinchilla dataset.
    assert comparison.validated is True
    assert comparison.reference_threshold_shift_db == pytest.approx(
      dataset.click_threshold_shift_db
    )
    assert comparison.reference_w1_ratio == pytest.approx(dataset.reference_w1_ratio)
    # Unchanged from before the loader refactor: the click condition, not the
    # tone average.
    assert comparison.reference_w1_ratio == pytest.approx(0.742, abs=0.01)

    # The simulated cohort preserves threshold and reproduces the attenuation.
    assert comparison.simulated_threshold_shift_db == pytest.approx(0.23, abs=0.05)
    assert comparison.simulated_w1_ratio == pytest.approx(46.6437 / 67.5298)
    assert comparison.threshold_shift_matched is True
    assert comparison.w1_ratio_matched is True

  def test_threshold_elevation_beyond_tolerance_fails(self):
    # Shift the exposed growth function far to the right of the baseline.
    elevated = dict(FULL_SWEEP_ABR_RESULTS)
    elevated["Selective-Synaptopathy"] = [0.0001, 0.0012, 0.0135, 0.1350, 4.5000, 46.6437]
    comparison = compare_to_empirical(FULL_SWEEP_CLICK_LEVELS, elevated)

    shift = comparison.simulated_threshold_shift_db
    assert shift is not None and shift > THRESHOLD_SHIFT_TOLERANCE_DB
    assert comparison.threshold_shift_matched is False
    assert comparison.w1_ratio_matched is True

  def test_ratio_outside_the_reference_range_fails(self):
    # Halve the suprathreshold response, well below the reference ratio.
    attenuated = dict(FULL_SWEEP_ABR_RESULTS)
    attenuated["Selective-Synaptopathy"] = [0.0123, 0.1238, 1.2915, 10.0933, 35.3127, 30.0]
    comparison = compare_to_empirical(FULL_SWEEP_CLICK_LEVELS, attenuated)

    ratio = comparison.simulated_w1_ratio
    assert ratio is not None
    assert abs(ratio - comparison.reference_w1_ratio) > W1_RATIO_TOLERANCE
    assert comparison.w1_ratio_matched is False

  def test_human_dataset_is_compared_without_a_pass_fail_verdict(self):
    # The cohorts are tuned to the chinchilla data, so a human comparison is
    # reported side by side rather than scored.
    human = load_abr_dataset("human")
    comparison = compare_to_empirical(
      FULL_SWEEP_CLICK_LEVELS, FULL_SWEEP_ABR_RESULTS, dataset=human
    )

    assert comparison.validated is False
    assert comparison.species == "human"
    assert comparison.comparison_group == "nexp"
    assert comparison.threshold_shift_matched is None
    assert comparison.w1_ratio_matched is None

    # Human data carries no ABR thresholds, so no threshold shift is claimed,
    # but the Wave-I ratio is still comparable.
    assert comparison.reference_threshold_shift_db is None
    assert comparison.reference_w1_ratio == pytest.approx(human.ratio_of_mean_w1)
    assert comparison.simulated_w1_ratio is not None

  def test_human_comparison_group_selects_its_own_reference(self):
    nexp = compare_to_empirical(
      FULL_SWEEP_CLICK_LEVELS, FULL_SWEEP_ABR_RESULTS, dataset=load_abr_dataset("human")
    )
    ma = compare_to_empirical(
      FULL_SWEEP_CLICK_LEVELS,
      FULL_SWEEP_ABR_RESULTS,
      dataset=load_abr_dataset("human", comparison_group="ma"),
    )
    assert ma.comparison_group == "ma"
    assert ma.reference_w1_ratio != pytest.approx(nexp.reference_w1_ratio)

  def test_human_table_omits_status_and_reports_no_threshold(self):
    comparison = compare_to_empirical(
      FULL_SWEEP_CLICK_LEVELS, FULL_SWEEP_ABR_RESULTS, dataset=load_abr_dataset("human")
    )
    table = format_empirical_comparison_table(comparison)
    assert "PASSED" not in table
    assert "FAILED" not in table
    assert "Status" not in table
    assert "not measured" in table
    assert "Exploratory comparison" in table

  def test_chinchilla_table_keeps_its_verdict(self):
    table = format_empirical_comparison_table(
      compare_to_empirical(FULL_SWEEP_CLICK_LEVELS, FULL_SWEEP_ABR_RESULTS)
    )
    assert "Status" in table
    assert "PASSED" in table

  def test_quick_sweep_leaves_metrics_unresolved(self):
    # The 60-80 dB SPL sweep never brackets the threshold criterion.
    comparison = compare_to_empirical([60.0, 80.0], {"Control": [10.7559, 67.5298]})

    assert comparison.simulated_threshold_shift_db is None
    assert comparison.threshold_shift_matched is None
    assert comparison.simulated_w1_ratio is None
    assert comparison.w1_ratio_matched is None

  def test_table_renders_values_and_unresolved_metrics(self):
    resolved = format_empirical_comparison_table(
      compare_to_empirical(FULL_SWEEP_CLICK_LEVELS, FULL_SWEEP_ABR_RESULTS)
    )
    assert "Click ABR threshold shift (dB)" in resolved
    assert "0.691" in resolved
    assert "PASSED" in resolved

    skipped = format_empirical_comparison_table(
      compare_to_empirical([60.0, 80.0], {"Control": [10.7559, 67.5298]})
    )
    assert "n/a" in skipped
    assert "SKIPPED" in skipped

  def test_plot_empirical_comparison(self, tmp_path: pathlib.Path):
    out = tmp_path / "empirical_comparison.png"
    dataset = load_abr_dataset()
    path = plot_empirical_comparison(
      compare_to_empirical(FULL_SWEEP_CLICK_LEVELS, FULL_SWEEP_ABR_RESULTS, dataset=dataset),
      out,
      dataset=dataset,
    )

    assert path == out
    assert out.exists()
    assert out.stat().st_size > 1000

  def test_plot_rejects_a_dataset_that_disagrees_with_the_comparison(self, tmp_path):
    human = compare_to_empirical(
      FULL_SWEEP_CLICK_LEVELS, FULL_SWEEP_ABR_RESULTS, dataset=load_abr_dataset("human")
    )
    with pytest.raises(ValueError, match="but the comparison describes"):
      plot_empirical_comparison(human, tmp_path / "mismatch.png", dataset=load_abr_dataset())
