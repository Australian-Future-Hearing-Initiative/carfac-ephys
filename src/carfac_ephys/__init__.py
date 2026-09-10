"""carfac-ephys: Simulating animal cochlear impairment electrophysiology with CARFAC."""

from carfac_ephys.carfac_model import (
  CarfacModel,
  FiberRetention,
  build_model,
)
from carfac_ephys.constants import (
  DEFAULT_SAMPLE_RATE,
  DYNAMIC_RANGE_DB,
  amplitude_to_db_spl,
  db_spl_to_amplitude,
)
from carfac_ephys.electrophysiology import (
  compute_population_rate,
  extract_efr_amplitude,
  extract_wave_i_amplitude,
)
from carfac_ephys.experiment import (
  BiologicalValidation,
  Cohort,
  CohortCondition,
  DEFAULT_CLICK_LEVELS_DB,
  DEFAULT_COHORT_CONDITIONS,
  DEFAULT_EFR_LEVELS_DB,
  format_markdown_table,
  generate_simulation_report,
  get_default_cohort,
  simulate_abr_level_series,
  simulate_efr_level_series,
  validate_biological_signatures,
)
from carfac_ephys.stimuli import generate_click, generate_sam_tone

__version__ = "0.1.0"

__all__ = [
  "BiologicalValidation",
  "CarfacModel",
  "Cohort",
  "CohortCondition",
  "DEFAULT_CLICK_LEVELS_DB",
  "DEFAULT_COHORT_CONDITIONS",
  "DEFAULT_EFR_LEVELS_DB",
  "DEFAULT_SAMPLE_RATE",
  "DYNAMIC_RANGE_DB",
  "FiberRetention",
  "amplitude_to_db_spl",
  "build_model",
  "compute_population_rate",
  "db_spl_to_amplitude",
  "extract_efr_amplitude",
  "extract_wave_i_amplitude",
  "format_markdown_table",
  "generate_click",
  "generate_sam_tone",
  "generate_simulation_report",
  "get_default_cohort",
  "simulate_abr_level_series",
  "simulate_efr_level_series",
  "validate_biological_signatures",
]
