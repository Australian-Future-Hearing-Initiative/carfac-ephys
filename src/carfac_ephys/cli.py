"""Command-line interface for running CARFAC electrophysiology cohort simulations."""

import pathlib

import click

from carfac_ephys import electrophysiology, empirical, experiment


@click.command(name="carfac-ephys-simulate")
@click.option(
  "--species",
  type=click.Choice(sorted(empirical.SPECIES_DATA_FILES), case_sensitive=False),
  default=empirical.DEFAULT_SPECIES,
  show_default=True,
  help="Empirical dataset to compare the simulated cohorts against.",
)
@click.option(
  "--comparison-group",
  default=None,
  help=(
    "Condition to compare against the dataset's baseline "
    "(human: 'nexp' or 'ma'). Defaults to the dataset's first comparison group."
  ),
)
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
  species: str,
  comparison_group: str | None,
  output_dir: pathlib.Path,
  plot: bool,
  quick: bool,
) -> None:
  """Runs CARFAC cochlear impairment electrophysiology cohort simulations."""
  # Load the dataset once, so every section below reports the same comparison
  # instead of each re-resolving its own default.
  dataset = empirical.load_abr_dataset(species.lower(), comparison_group=comparison_group)
  slug = experiment.dataset_slug(dataset)
  click.echo(f"Dataset: {dataset.species} ({dataset.comparison_group} vs {dataset.baseline_label})")

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
  click.echo(experiment.format_calibration_line(click_levels, abr_results, dataset=dataset) + "\n")

  # Quantify the agreement with the empirical measurements.
  comparison = experiment.compare_to_empirical(click_levels, abr_results, dataset=dataset)
  verdict = "Validation" if comparison.validated else "Exploratory comparison"
  click.echo(f"{verdict} against the {dataset.species} dataset ({dataset.source}):")
  click.echo(experiment.format_empirical_comparison_table(comparison) + "\n")

  # Generate and save diagnostic figures if requested.
  if plot:
    output_dir.mkdir(parents=True, exist_ok=True)

    abr_fig_path = output_dir / "abr_wave_i_growth.png"
    experiment.plot_abr_growth(click_levels, abr_results, abr_fig_path)
    click.echo(f"Saved ABR Wave-I growth figure to: {abr_fig_path}")

    efr_fig_path = output_dir / "efr_growth.png"
    experiment.plot_efr_growth(efr_levels, efr_results, efr_fig_path)
    click.echo(f"Saved EFR growth figure to: {efr_fig_path}")

    empirical_fig_path = output_dir / f"empirical_comparison_{slug}.png"
    experiment.plot_empirical_comparison(comparison, empirical_fig_path, dataset=dataset)
    click.echo(f"Saved empirical comparison figure to: {empirical_fig_path}")

    report_path = output_dir / f"simulation_report_{slug}.md"
    experiment.generate_simulation_report(
      click_levels_db=click_levels,
      abr_results=abr_results,
      efr_levels_db=efr_levels,
      efr_results=efr_results,
      dataset=dataset,
      output_path=report_path,
    )
    click.echo(f"Saved simulation report to: {report_path}")


if __name__ == "__main__":
  main()
