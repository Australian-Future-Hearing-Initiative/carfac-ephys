"""Command-line interface for running CARFAC electrophysiology cohort simulations."""

import pathlib

import click

from carfac_ephys import electrophysiology, empirical, experiment
from carfac_ephys import unitary_response as unitary_response_module
from carfac_ephys.constants import DEFAULT_SAMPLE_RATE
from carfac_ephys.middle_ear import FILTERS as _MIDDLE_EAR_FILTERS


def _comparison_group_help() -> str:
  """Builds the help text from the datasets themselves, so it cannot drift."""
  options = "; ".join(
    f"{species}: {', '.join(repr(label) for label in files.comparison_labels)}"
    for species, files in sorted(empirical.SPECIES_DATA_FILES.items())
  )
  return (
    f"Condition to compare against the dataset's baseline ({options}). "
    "Defaults to the dataset's first comparison group."
  )


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
  help=_comparison_group_help(),
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
@click.option(
  "--middle-ear",
  type=click.Choice(["none", *sorted(_MIDDLE_EAR_FILTERS)], case_sensitive=False),
  default="none",
  show_default=True,
  help=(
    "Middle-ear filter applied to each stimulus before the cochlear model. "
    "'none' reproduces the published results; the calibration was fitted without one."
  ),
)
@click.option(
  "--unitary-response",
  type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
  default=None,
  help=(
    "Path to a unitary response kernel (.npy or one-column text). Convolves the "
    "compound AN activity with it to form a far-field potential. Omit to measure "
    "Wave-I from the population rate, which is what the published results use."
  ),
)
@click.option(
  "--unitary-response-rate",
  type=int,
  default=DEFAULT_SAMPLE_RATE,
  show_default=True,
  help="Sampling rate the UR file is defined at, in Hz. Resampled if it differs.",
)
def main(
  species: str,
  comparison_group: str | None,
  output_dir: pathlib.Path,
  plot: bool,
  quick: bool,
  middle_ear: str,
  unitary_response: pathlib.Path | None,
  unitary_response_rate: int,
) -> None:
  """Runs CARFAC cochlear impairment electrophysiology cohort simulations."""
  # Load the dataset once, so every section below reports the same comparison
  # instead of each re-resolving its own default.
  try:
    dataset = empirical.load_abr_dataset(
      species.lower(),
      comparison_group=None if comparison_group is None else comparison_group.lower(),
    )
  except (KeyError, ValueError) as error:
    raise click.BadParameter(str(error), param_hint="--comparison-group") from error
  slug = experiment.dataset_slug(dataset)
  click.echo(f"Dataset: {dataset.species} ({dataset.comparison_group} vs {dataset.baseline_label})")

  # Add the middle ear filter
  middle_ear_filter = None if middle_ear.lower() == "none" else _MIDDLE_EAR_FILTERS[middle_ear]
  if middle_ear_filter is not None:
    click.echo(f"Middle-ear filter: {middle_ear_filter.name}")

  # Determine stimulus levels.
  click_levels = [60.0, 80.0] if quick else list(experiment.DEFAULT_CLICK_LEVELS_DB)
  efr_levels = [60.0, 80.0] if quick else list(experiment.DEFAULT_EFR_LEVELS_DB)

  # Run ABR Wave-I level series simulation.
  click.echo("Running ABR Wave-I click level series simulation...")
  abr_results = experiment.simulate_abr_level_series(
    click_levels_db=click_levels, middle_ear_filter=middle_ear_filter
  )
  abr_table = experiment.format_ascii_table(
    title=f"ABR Wave-I Onset Amplitude ({electrophysiology.RESPONSE_UNIT}) vs Sound Level",
    levels_db=click_levels,
    results=abr_results,
  )
  click.echo("\n" + abr_table + "\n")

  # Run EFR SAM tone level series simulation.
  click.echo("Running EFR SAM tone level series simulation...")
  efr_results = experiment.simulate_efr_level_series(
    efr_levels_db=efr_levels, middle_ear_filter=middle_ear_filter
  )
  efr_table = experiment.format_ascii_table(
    title=f"EFR Spectral Magnitude at 100 Hz ({electrophysiology.RESPONSE_UNIT}) vs Sound Level",
    levels_db=efr_levels,
    results=efr_results,
  )
  click.echo("\n" + efr_table + "\n")

  # Unitary response
  response = None
  if unitary_response is not None:
    response = unitary_response_module.load(
      unitary_response, unitary_response_rate, source=str(unitary_response)
    )
    click.echo(f"Unitary response: {response.source} ({response.duration_s * 1000:.1f} ms)")

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
