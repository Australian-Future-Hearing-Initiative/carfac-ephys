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
from carfac_ephys.stimuli import generate_click, generate_sam_tone

__version__ = "0.1.0"

__all__ = [
  "CarfacModel",
  "DEFAULT_SAMPLE_RATE",
  "DYNAMIC_RANGE_DB",
  "FiberRetention",
  "amplitude_to_db_spl",
  "build_model",
  "compute_population_rate",
  "db_spl_to_amplitude",
  "extract_efr_amplitude",
  "extract_wave_i_amplitude",
  "generate_click",
  "generate_sam_tone",
]
