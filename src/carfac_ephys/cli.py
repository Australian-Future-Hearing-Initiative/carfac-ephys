"""Command-line interface for running CARFAC electrophysiology cohort simulations."""

import pathlib

import click

from carfac_ephys import electrophysiology, experiment


@click.command(name="carfac-ephys-simulate")
@click.option(
  "--output-dir",
  type=click.Path(path_type=pathlib.Path),
  default="output",
  show_default=True,
  help="Directory where output figures and tables are saved.",
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
  plot: bool,
  quick: bool,
) -> None:
  """Runs CARFAC cochlear impairment electrophysiology cohort simulations."""
  # Determine stimulus levels.
  click_levels = [60.0, 80.0] if quick else list(experiment.DEFAULT_CLICK_LEVELS_DB)
  efr_levels = [60.0, 80.0] if quick else list(experiment.DEFAULT_EFR_LEVELS_DB)

  # Run ABR Wave-I level series simulation.
  click.echo("Running ABR Wave-I click level series simulation...")
  abr_results = experiment.simulate_abr_level_series(click_levels_db=click_levels)
  abr_table = experiment.format_ascii_table(
    title=f"ABR Wave-I Onset Amplitude ({electrophysiology.RESPONSE_UNIT}) vs Sound Level",
    levels_db=click_levels,
    results=abr_results,
  )
  click.echo("\n" + abr_table + "\n")

  # Run EFR SAM tone level series simulation.
  click.echo("Running EFR SAM tone level series simulation...")
  efr_results = experiment.simulate_efr_level_series(efr_levels_db=efr_levels)
  efr_table = experiment.format_ascii_table(
    title=f"EFR Spectral Magnitude at 100 Hz ({electrophysiology.RESPONSE_UNIT}) vs Sound Level",
    levels_db=efr_levels,
    results=efr_results,
  )
  click.echo("\n" + efr_table + "\n")

  # Report the fitted conversion from arbitrary units to microvolts.
  click.echo(experiment.format_calibration_line(click_levels, abr_results) + "\n")

  # Generate and save diagnostic figures if requested.
  if plot:
    output_dir.mkdir(parents=True, exist_ok=True)

    abr_fig_path = output_dir / "abr_wave_i_growth.png"
    experiment.plot_abr_growth(click_levels, abr_results, abr_fig_path)
    click.echo(f"Saved ABR Wave-I growth figure to: {abr_fig_path}")

    efr_fig_path = output_dir / "efr_growth.png"
    experiment.plot_efr_growth(efr_levels, efr_results, efr_fig_path)
    click.echo(f"Saved EFR growth figure to: {efr_fig_path}")

    report_path = output_dir / "simulation_report.md"
    experiment.generate_simulation_report(
      click_levels_db=click_levels,
      abr_results=abr_results,
      efr_levels_db=efr_levels,
      efr_results=efr_results,
      output_path=report_path,
    )
    click.echo(f"Saved simulation report to: {report_path}")


if __name__ == "__main__":
  main()
