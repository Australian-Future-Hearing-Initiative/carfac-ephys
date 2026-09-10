# CARFAC Cochlear Impairment Electrophysiology (`carfac-ephys`)

[![CI Tests](https://github.com/Australian-Future-Hearing-Initiative/carfac-ephys/actions/workflows/test.yml/badge.svg)](https://github.com/Australian-Future-Hearing-Initiative/carfac-ephys)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

In silico reproduction of animal model cochlear impairment electrophysiology using the [CARFAC](https://github.com/google/carfac) (Cascade of Asymmetric Resonators with Fast-Acting Compression) auditory periphery model.

This repository demonstrates that CARFAC accurately reproduces the signature electrophysiological dissociations observed in animal models (e.g., chinchilla and mouse studies; Bharadwaj et al. 2022, Mehraei et al. 2016, Ginsberg et al. 2023) across Auditory Brainstem Response (ABR) Wave-I / Compound Action Potential (CAP) and Envelope-Following Response (EFR) level series.

---

## 1. Overview & Biological Background

In animal hearing loss studies, controlled cochlear pathologies produce distinct electrophysiological signatures:

1. **Cochlear Synaptopathy / Auditory Neuropathy** (loss of auditory nerve synapses/fibers with intact hair cells):
   - **Animal Hallmark:** Detection thresholds at low sound levels are preserved; suprathreshold ABR Wave-I amplitudes and EFR phase-locking/magnitudes are attenuated proportionally to fiber loss.
2. **Outer Hair Cell (OHC) Loss / Sensory Hearing Loss** (stereocilia and active gain loss):
   - **Animal Hallmark:** Marked threshold elevation (20–40 dB), rightward-shifted input-output growth curves, and loss of compressive nonlinearity (steep recruitment).
3. **Mixed Pathology**:
   - **Animal Hallmark:** Combined threshold elevation and reduced suprathreshold response ceiling.

`carfac-ephys` reproduces these physiological responses by systematically manipulating CARFAC's biophysical parameters:
- `ohc_health` $\in [0, 1]$: Scales outer hair cell motility and active basilar membrane undamping.
- `fiber_retention` $\in [0, 1]$: Scales auditory nerve fiber counts across spontaneous rate groups (HSR, MSR, LSR).

---

## 2. Experimental Findings

Simulated cohort responses across five representative conditions:
- **Control**: Healthy cochlea ($100\%$ OHC health, $100\%$ AN fibers).
- **Synaptopathy-50**: Moderate synaptopathy ($100\%$ OHC health, $50\%$ AN fibers).
- **Synaptopathy-25**: Severe synaptopathy ($100\%$ OHC health, $25\%$ AN fibers).
- **OHC-Loss**: Sensory hearing loss ($40\%$ OHC health, $100\%$ AN fibers).
- **Mixed-Loss**: Combined pathology ($40\%$ OHC health, $50\%$ AN fibers).

### ABR Wave-I Onset Amplitude (Clicks: 30–80 dB SPL)

Responses are in arbitrary units (AU): CARFAC neural activity patterns are dimensionless model output, not calibrated firing rates or recorded voltages. `carfac_ephys.fit_response_scale_uv_per_au` fits the conversion to microvolts by matching the Control response at 80 dB SPL to the pre-exposure chinchilla click Wave-I amplitude of Bharadwaj et al. (2022), giving $\approx 0.0316$ $\mu$V/AU.

| Condition | 30 dB SPL | 40 dB SPL | 50 dB SPL | 60 dB SPL | 70 dB SPL | 80 dB SPL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 0.0128 | 0.1282 | 1.3399 | 10.7559 | 51.5883 | **67.5298** |
| **Synaptopathy-50** | 0.0075 | 0.0760 | 0.7949 | 6.1125 | 28.1907 | **35.5266** *(52.6% of Control)* |
| **Synaptopathy-25** | 0.0041 | 0.0414 | 0.4330 | 3.2536 | 14.6364 | **18.2008** *(27.0% of Control)* |
| **OHC-Loss** | 0.0008 | 0.0025 | 0.0078 | 0.0256 | 0.1077 | **0.8301** *(~30 dB threshold shift)* |
| **Mixed-Loss** | 0.0004 | 0.0012 | 0.0039 | 0.0128 | 0.0547 | **0.4261** |

![ABR Wave-I Growth Curves](assets/abr_wave_i_growth.png)

### Envelope Following Response (EFR) Growth (SAM Tones: 40–80 dB SPL)

Carrier $f_c = 2000$ Hz, modulation frequency $f_m = 100$ Hz, $100\%$ modulation depth:

| Condition | 40 dB SPL | 50 dB SPL | 60 dB SPL | 70 dB SPL | 80 dB SPL |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control** | 1.0718 | 1.7686 | 2.5543 | 3.6773 | **5.9665** |
| **Synaptopathy-50** | 0.7172 | 1.0091 | 1.5053 | 2.0680 | **2.9614** *(49.6% of Control)* |
| **Synaptopathy-25** | 0.3906 | 0.6391 | 0.7122 | 1.2044 | **1.9453** *(32.6% of Control)* |
| **OHC-Loss** | 0.0038 | 0.0382 | 0.3602 | 1.8413 | **4.9530** |
| **Mixed-Loss** | 0.0023 | 0.0230 | 0.2180 | 1.0620 | **2.6369** |

![EFR Growth Curves](assets/efr_growth.png)

---

## 3. Reproduction Steps

### Prerequisites
- Python $\ge 3.11$
- [`uv`](https://github.com/astral-sh/uv) package manager

### 1. Installation

Clone and install dependencies with `uv`:
```bash
git clone https://github.com/Australian-Future-Hearing-Initiative/carfac-ephys.git
cd carfac-ephys
uv sync
```

### 2. Run Automated Test Suite

Verify all 92 unit and integration tests:
```bash
uv run pytest -v
```

### 3. Run Cohort Simulation

Execute the full level sweep across all five cohorts:
```bash
uv run carfac-ephys-simulate --output-dir output/
```

This will:
1. Run click-evoked ABR simulations from 30 to 80 dB SPL.
2. Run SAM-tone EFR simulations from 40 to 80 dB SPL.
3. Validate all biological signatures (assertions matching animal literature).
4. Print summary tables to stdout.
5. Save diagnostic figures (`abr_wave_i_growth.png`, `efr_growth.png`) and `simulation_report.md` to `output/`.

#### Fast Smoke Test
To verify the pipeline on a reduced 2-level subset:
```bash
uv run carfac-ephys-simulate --quick --output-dir output/
```

---

## 4. Package Architecture

```text
carfac-ephys/
├── pyproject.toml              # Standalone dependencies and scripts
├── README.md                   # Documentation and walkthrough
├── assets/                     # Published diagnostic figures
│   ├── abr_wave_i_growth.png
│   └── efr_growth.png
├── src/
│   └── carfac_ephys/
│       ├── __init__.py
│       ├── constants.py        # Calibration constants (104 dB SPL = 0 dB FS)
│       ├── stimuli.py          # Calibrated click and SAM tone generation
│       ├── carfac_model.py     # Biophysical CARFAC wrapper (OHC & fiber retention)
│       ├── electrophysiology.py # ABR Wave-I and EFR metric extractors
│       ├── experiment.py       # Cohort definitions, level sweeps, validation
│       └── cli.py              # CLI entry point (carfac-ephys-simulate)
└── tests/
    ├── test_stimuli.py
    ├── test_carfac_model.py
    ├── test_electrophysiology.py
    └── test_experiment.py
```

---

## 5. References

- **Bharadwaj HM, Hustedt-Mai AR, Ginsberg HM, et al.** (2022). *Cross-species experiments reveal widespread cochlear neural damage in normal hearing.* Communications Biology, 5(1), 733. [doi:10.1038/s42003-022-03691-4](https://doi.org/10.1038/s42003-022-03691-4).
- **Mehraei G, Hickox AE, Bharadwaj HM, et al.** (2016). *Auditory Brainstem Response Latency in Noise as a Marker of Cochlear Synaptopathy.* Journal of Neuroscience, 36(13), 3755–3764. [doi:10.1523/JNEUROSCI.4460-15.2016](https://doi.org/10.1523/JNEUROSCI.4460-15.2016).
- **Ginsberg HM, Singh R, Bharadwaj HM, Heinz MG.** (2023). *A multi-channel EEG mini-cap can improve reliability for recording auditory brainstem responses in chinchillas.* Journal of Neuroscience Methods, 398, 109954. [doi:10.1016/j.jneumeth.2023.109954](https://doi.org/10.1016/j.jneumeth.2023.109954).
- **Dolphin WF, Mountain DC.** (1992). *The envelope following response: Scalp potentials elicited in the Mongolian gerbil using sinusoidally AM acoustic signals.* J. Acoust. Soc. Am., 92(1), 86–96. [doi:10.1121/1.404225](https://doi.org/10.1121/1.404225).
- **Lyon RF.** (2017). *Human and Machine Hearing: Extracting Meaning from Sound.* Cambridge University Press.
