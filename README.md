# carfac-ephys

Reproducing animal model cochlear impairment electrophysiology (ABR Wave-I and EFR level series) in silico using the CARFAC model.

## Features

- **Biophysical Impairments**: Outer hair cell (OHC) loss (`ohc_health`) and auditory nerve synaptopathy / deafferentation (`fiber_retention`).
- **Calibrated Stimulus Suite**: Clicks (100 us pulse, 30-80 dB SPL) and SAM tones (fc=2000 Hz, fm=100 Hz, 40-80 dB SPL) calibrated against CARFAC digital reference (104 dB SPL = 0 dB FS).
- **Electrophysiological Response Extraction**: Population auditory nerve firing rate, ABR Wave-I onset amplitude, and Envelope Following Response (EFR) spectral magnitude at fm.
- **Animal Literature Reproduction**:
  - *Synaptopathy* (Bharadwaj et al. 2022, Mehraei et al. 2016): preserved low-level thresholds (30-40 dB SPL), proportional high-level Wave-I and EFR amplitude reduction.
  - *OHC Loss* (Ruggero et al. 1997): elevated threshold (~30 dB shift), loss of compressive amplification.
  - *Mixed Loss*: dual deficit combining elevated threshold and attenuated maximum output.

## Installation & CLI Usage

Run simulation across cohorts and generate publication figures and report:

```bash
uv run carfac-ephys-simulate --output-dir output/
```

Options:
- `--output-dir PATH`: Directory where figures and simulation report are saved (default: `output/`).
- `--plot / --no-plot`: Enable/disable diagnostic plot and report generation (default: `--plot`).
- `--quick / --full`: Fast 2-level test sweep vs full level series (default: `--full`).

## Testing

```bash
uv run pytest tests/
```
