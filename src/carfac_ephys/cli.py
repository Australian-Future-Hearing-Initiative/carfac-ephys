"""Command-line interface for running CARFAC electrophysiology cohort simulations."""

import pathlib

import click

from carfac_ephys import electrophysiology, empirical, experiment


@click.command(name="carfac-ephys-simulate")
@click.option(
  "--output-dir",
  type=click.Path(path_type=pathlib.Path),
  default="output",
  show_default=True,
  help="Directory where output figures and tables are saved.",
)
@click.option(
  "--stimulus",
  type=click.Choice(["all", "click", "tone-burst"], case_sensitive=False),
  default="all",
  show_default=True,
  help="Type of stimulus to simulate: 'all', 'click', or 'tone-burst'.",
)
@click.option(
  "--plot/--no-plot",
  default=True,
  show_default=True,
  help="Generate and save publication-quality figures.",
)
@click.option(
  "--quick/--full",
  default=False,
  help="Run a fast 2-level sweep instead of the full level series.",
)
def main(
  output_dir: pathlib.Path,
  stimulus: str,
  plot: bool,
  quick: bool,
) -> None:
  """Runs CARFAC cochlear impairment electrophysiology cohort simulations."""
  stimulus_mode = stimulus.lower()
  run_click = stimulus_mode in ("all", "click")
  run_tone_burst = stimulus_mode in ("all", "tone-burst")

  if plot:
    output_dir.mkdir(parents=True, exist_ok=True)

  # --- Broadband Click & EFR Simulations ---
  if run_click:
    click_levels = [60.0, 80.0] if quick else list(experiment.DEFAULT_CLICK_LEVELS_DB)
    efr_levels = [60.0, 80.0] if quick else list(experiment.DEFAULT_EFR_LEVELS_DB)

    click.echo("Running ABR Wave-I click level series simulation...")
    abr_results = experiment.simulate_abr_level_series(click_levels_db=click_levels)
    abr_table = experiment.format_ascii_table(
      title=f"ABR Wave-I Onset Amplitude ({electrophysiology.RESPONSE_UNIT}) vs Sound Level (Broadband Click)",
      levels_db=click_levels,
      results=abr_results,
    )
    click.echo("\n" + abr_table + "\n")

    click.echo("Running EFR SAM tone level series simulation...")
    efr_results = experiment.simulate_efr_level_series(efr_levels_db=efr_levels)
    efr_table = experiment.format_ascii_table(
      title=f"EFR Spectral Magnitude at 100 Hz ({electrophysiology.RESPONSE_UNIT}) vs Sound Level",
      levels_db=efr_levels,
      results=efr_results,
    )
    click.echo("\n" + efr_table + "\n")

    # Report the fitted conversion from arbitrary units to microvolts for clicks.
    click.echo(
      experiment.format_calibration_line(
        click_levels, abr_results, frequency_hz=empirical.CLICK_FREQUENCY_HZ
      )
      + "\n"
    )

    # Quantify the agreement with the chinchilla ABR measurements.
    comparison = experiment.compare_to_empirical(click_levels, abr_results)
    click.echo("Empirical comparison against Bharadwaj et al. (2022):")
    click.echo(experiment.format_empirical_comparison_table(comparison) + "\n")

    if plot:
      abr_fig_path = output_dir / "abr_wave_i_growth.png"
      experiment.plot_abr_growth(
        click_levels, abr_results, abr_fig_path, stimulus_label="Broadband Click"
      )
      click.echo(f"Saved ABR Wave-I growth figure to: {abr_fig_path}")

      abr_click_fig_path = output_dir / "abr_wave_i_growth_click.png"
      experiment.plot_abr_growth(
        click_levels, abr_results, abr_click_fig_path, stimulus_label="Broadband Click"
      )
      click.echo(f"Saved ABR Wave-I click growth figure to: {abr_click_fig_path}")

      efr_fig_path = output_dir / "efr_growth.png"
      experiment.plot_efr_growth(efr_levels, efr_results, efr_fig_path)
      click.echo(f"Saved EFR growth figure to: {efr_fig_path}")

      empirical_fig_path = output_dir / "empirical_comparison.png"
      experiment.plot_empirical_comparison(comparison, empirical_fig_path)
      click.echo(f"Saved empirical comparison figure to: {empirical_fig_path}")

      report_path = output_dir / "simulation_report.md"
      experiment.generate_simulation_report(
        click_levels_db=click_levels,
        abr_results=abr_results,
        efr_levels_db=efr_levels,
        efr_results=efr_results,
        output_path=report_path,
      )
      click.echo(f"Saved simulation report to: {report_path}\n")

  # --- Tone Burst Simulations (4 kHz & 8 kHz) ---
  if run_tone_burst:
    tb_levels = [60.0, 80.0] if quick else list(experiment.DEFAULT_TONE_BURST_LEVELS_DB)

    click.echo(
      "Running ABR Wave-I tone-burst level series simulation (4 kHz & 8 kHz, alternating polarity)..."
    )
    tb_results = experiment.simulate_tone_burst_cohort(tone_burst_levels_db=tb_levels)

    for freq_hz in (4000.0, 8000.0):
      table = experiment.format_ascii_table(
        title=(
          f"ABR Wave-I Onset Amplitude ({electrophysiology.RESPONSE_UNIT}) vs"
          f" Sound Level ({freq_hz / 1000:g} kHz Tone Burst)"
        ),
        levels_db=tb_levels,
        results=tb_results.results_by_frequency.get(freq_hz, {}),
      )
      click.echo("\n" + table)

    avg_table = experiment.format_ascii_table(
      title=(
        f"ABR Wave-I Onset Amplitude ({electrophysiology.RESPONSE_UNIT}) vs"
        " Sound Level (4/8 kHz Composite Average)"
      ),
      levels_db=tb_levels,
      results=tb_results.composite_results,
    )
    click.echo("\n" + avg_table + "\n")

    # Tone burst calibrations
    cal_line_4k = experiment.format_calibration_line(
      tb_levels, tb_results.results_4k, frequency_hz=4000.0
    )
    cal_line_8k = experiment.format_calibration_line(
      tb_levels, tb_results.results_8k, frequency_hz=8000.0
    )
    cal_line_avg = experiment.format_calibration_line(
      tb_levels, tb_results.composite_results, frequency_hz=None
    )
    click.echo(cal_line_4k)
    click.echo(cal_line_8k)
    click.echo(cal_line_avg + "\n")

    # Quantify tone-burst empirical agreement
    tb_comparison = experiment.compare_tone_burst_to_empirical(tb_results)
    click.echo("Empirical comparison against Bharadwaj et al. (2022) [Tone Bursts]:")
    click.echo(experiment.format_tone_burst_comparison_table(tb_comparison) + "\n")

    if plot:
      click.echo("Simulating tone-burst waveforms at 80 dB SPL for visualization...")
      waveforms_4k = experiment.simulate_tone_burst_waveforms(
        frequency_hz=4000.0, level_db=80.0
      )
      waveforms_8k = experiment.simulate_tone_burst_waveforms(
        frequency_hz=8000.0, level_db=80.0
      )

      wave_fig_path = output_dir / "tone_burst_waveforms.png"
      experiment.plot_tone_burst_waveforms(
        waveforms_4k, waveforms_8k, wave_fig_path, level_db=80.0
      )
      click.echo(f"Saved tone-burst response waveforms figure to: {wave_fig_path}")

      tb_growth_fig_path = output_dir / "abr_wave_i_growth_tone_burst.png"
      experiment.plot_tone_burst_growth(tb_results, tb_growth_fig_path)
      click.echo(f"Saved tone-burst ABR Wave-I growth figure to: {tb_growth_fig_path}")


if __name__ == "__main__":
  main()
