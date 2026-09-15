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

---

## 2. How the Synthetic Model Replicates Animal Data

The core objective of `carfac-ephys` is reproducing the empirical electrophysiological findings of animal synaptopathy studies (specifically **Bharadwaj et al. 2022**, *Commun Biol*, investigating noise-exposed chinchillas) using the CARFAC biophysical cochlear model.

### 2.1 The Animal Experiment Being Replicated

In the Bharadwaj et al. (2022) chinchilla study:
1. **Noise Exposure**: Chinchillas were exposed to an acoustic overexposure that induced temporary threshold shifts (TTS).
2. **Permanent Synaptopathy with Intact Hair Cells**: After 2 weeks, outer hair cells fully recovered (confirmed by normal audiometric thresholds and DPOAEs), but auditory nerve ribbon synapses suffered permanent loss. Crucially, noise damage preferentially destroys low- and medium-spontaneous-rate (LSR/MSR) fibers while sparing high-spontaneous-rate (HSR) fibers.
3. **Electrophysiological Hallmarks**:
   - **Thresholds Preserved**: Click ABR threshold shift was negligible ($-0.30$ dB measured).
   - **Suprathreshold Wave-I Reduced**: At 80 dB SPL, ABR Wave-I amplitude dropped to $74.2\%$ of pre-exposure baseline ($25.8\%$ attenuation).

### 2.2 In Silico Simulation Pipeline

The synthetic pipeline bridges acoustic input to simulated electrophysiology through CARFAC:

```text
Acoustic Stimulus (Clicks / SAM Tones)
       │
       ▼
CARFAC Cochlear Resonators (Basilar Membrane Filterbank)
       │  [Modulated by ohc_health: active nonlinear gain]
       ▼
Inner Hair Cell Synapse Model (Two-Capacitor Reservoir)
       │  [Modulated by fiber_retention: HSR, MSR, LSR counts]
       ▼
Neural Activity Patterns: naps(t, channel)
       │
       ▼  Population Rate Summation: r(t) = Σ_c naps(t, c)
Compound Action Potential (CAP) Waveform
       │
       ▼  Wave-I Peak Extraction: max r(t) - baseline (0–8 ms window)
Simulated Response (Arbitrary Units, AU)
       │
       ▼  Scale Factor: α = W1_animal_pre_80dB / Wave-I_sim_ctrl_80dB ≈ 0.0316 μV/AU
Calibrated Electrophysiology (μV) & Threshold Interpolation (0.1 μV criterion)
```

### 2.3 Parameter Mapping: Biology to CARFAC

To model the animal cohorts in silico, CARFAC parameters are configured as follows:

| Cohort | Biological Pathology | `ohc_health` | `fiber_retention` `(HSR, MSR, LSR)` | Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **Control** | Healthy baseline | `1.0` | `(1.0, 1.0, 1.0)` | Full OHC amplification and 100% nerve fibers. |
| **Selective-Synaptopathy** | Noise-exposed animal model | `1.0` | `(1.0, 0.5, 0.0)` | Replicates noise synaptopathy: intact hair cells (`ohc=1.0`), spared sensitive HSR fibers (`100%`), and depleted MSR/LSR fibers (`50%` / `0%`). |
| **Synaptopathy-50** | Uniform deafferentation | `1.0` | `0.5` `(0.5, 0.5, 0.5)` | Uniform 50% fiber loss across all spontaneous rate groups. |
| **Synaptopathy-25** | Severe deafferentation | `1.0` | `0.25` `(0.25, 0.25, 0.25)` | Uniform 75% fiber loss across all spontaneous rate groups. |
| **OHC-Loss** | Sensory hair cell loss | `0.4` | `(1.0, 1.0, 1.0)` | Basilar membrane undamping loss (20–30 dB threshold elevation). |
| **Mixed-Loss** | Combined sensory & neural | `0.4` | `0.5` `(0.5, 0.5, 0.5)` | Combined OHC active gain loss and neural deafferentation. |

### 2.4 Biophysical Mechanism: Why the Model Matches Animal Data

1. **Near-Threshold Sparing ($30–40$ dB SPL)**:
   - Near threshold, sound pressures only activate the most sensitive fibers: high-spontaneous-rate (HSR) fibers.
   - Because `Selective-Synaptopathy` retains $100\%$ of HSR fibers, the simulated response near threshold is $96.5\%$ of Control (versus only $59.3\%$ for uniform deafferentation).
   - As a result, the simulated click threshold shift is $+0.23$ dB, closely replicating the animal measurement ($-0.30$ dB, both effectively 0 dB).
2. **Suprathreshold Attenuation ($70–80$ dB SPL)**:
   - At high sound levels, HSR fibers saturate. Further growth in population firing rate requires recruitment of high-threshold, low- and medium-spontaneous-rate (LSR/MSR) fibers.
   - Because MSR/LSR fibers are depleted ($50\%$ MSR, $0\%$ LSR), the suprathreshold growth plateaus early, yielding a simulated post/pre Wave-I ratio of $0.691$ at 80 dB SPL (matching the animal measurement of $0.742 \pm 0.10$).

---

## 3. Experimental Findings

Simulated cohort responses across six representative conditions:
- **Control**: Healthy cochlea ($100\%$ OHC health, $100\%$ AN fibers).
- **Synaptopathy-50**: Moderate synaptopathy ($100\%$ OHC health, $50\%$ AN fibers).
- **Synaptopathy-25**: Severe synaptopathy ($100\%$ OHC health, $25\%$ AN fibers).
- **Selective-Synaptopathy**: Selective loss of high-threshold fibers ($100\%$ OHC health, $100\%$ HSR, $50\%$ MSR, $0\%$ LSR).
- **OHC-Loss**: Sensory hearing loss ($40\%$ OHC health, $100\%$ AN fibers).
- **Mixed-Loss**: Combined pathology ($40\%$ OHC health, $50\%$ AN fibers).

### 3.1 Broadband Click Data

#### ABR Wave-I Onset Amplitude (Broadband Clicks: 30–80 dB SPL)

Responses are in arbitrary units (AU): CARFAC neural activity patterns are dimensionless model output, not calibrated firing rates or recorded voltages. `carfac_ephys.fit_response_scale_uv_per_au` fits the conversion to microvolts by matching the Control response at 80 dB SPL to the pre-exposure chinchilla click Wave-I amplitude of Bharadwaj et al. (2022), giving $\approx 0.0316$ $\mu$V/AU.

| Condition | 30 dB SPL | 40 dB SPL | 50 dB SPL | 60 dB SPL | 70 dB SPL | 80 dB SPL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 0.0128 | 0.1282 | 1.3399 | 10.7559 | 51.5884 | **67.5298** |
| **Synaptopathy-50** | 0.0075 | 0.0760 | 0.7949 | 6.1125 | 28.1906 | **35.5266** *(52.6% of Control)* |
| **Synaptopathy-25** | 0.0041 | 0.0414 | 0.4330 | 3.2536 | 14.6364 | **18.2007** *(27.0% of Control)* |
| **Selective-Synaptopathy** | 0.0123 | 0.1238 | 1.2915 | 10.0933 | 35.3128 | **46.6437** *(69.1% of Control)* |
| **OHC-Loss** | 0.0008 | 0.0025 | 0.0078 | 0.0256 | 0.1077 | **0.8301** *(~30 dB threshold shift)* |
| **Mixed-Loss** | 0.0004 | 0.0012 | 0.0039 | 0.0128 | 0.0547 | **0.4261** |

Selective loss of the high-threshold fibers leaves the near-threshold response almost intact ($96.5\%$ of Control at 40 dB SPL, versus $59.3\%$ for uniform $50\%$ deafferentation) while still cutting the suprathreshold Wave-I to $69.1\%$ — the hidden-hearing-loss signature, and within the $0.74 \pm 0.10$ post/pre Wave-I ratio measured in noise-exposed chinchillas (Bharadwaj et al. 2022).

![ABR Wave-I Growth Curves (Broadband Click)](assets/abr_wave_i_growth_click.png)

#### Empirical Validation Against Chinchilla Click ABR Data

The Selective-Synaptopathy cohort stands in for the noise-exposed chinchillas of Bharadwaj et al. (2022), which recovered their click ABR thresholds two weeks after exposure while retaining a reduced suprathreshold Wave-I. Simulated thresholds are the $0.1$ $\mu$V crossing of the interpolated Wave-I growth function, converted through the fitted scale factor; the animal values come from the packaged dataset (`carfac_ephys.load_chinchilla_abr_dataset`), not from hand-picked bands.

| Metric | Simulated (Selective-Synaptopathy) | Animal (Bharadwaj et al. 2022) | Tolerance | Status |
| :--- | :---: | :---: | :---: | :---: |
| Click ABR threshold shift (dB) | $+0.23$ | $-0.30$ | $\pm 3$ dB | PASSED |
| Suprathreshold Wave-I post/pre ratio (80 dB SPL) | $0.691$ | $0.742$ | $\pm 0.10$ | PASSED |

![Simulated versus Animal ABR Comparison](assets/empirical_comparison.png)

---

### 3.2 Tone-Burst Data (4 kHz & 8 kHz)

Tone burst stimuli (5 ms duration, 0.5 ms linear rise/fall ramps, 20 Hz stimulation rate, 500 repetitions) are simulated across 30 to 80 dB SPL at 4 kHz and 8 kHz with alternating polarities ($+1.0$ and $-1.0$). Averaging the positive and negative polarity responses cancels the phase-locked cochlear microphonic (CM) and stimulus artifact, isolating the rectified neural compound action potential (Wave-I).

#### 4 kHz Tone-Burst ABR Wave-I (30–80 dB SPL)
`carfac_ephys.fit_response_scale_uv_per_au(..., frequency_hz=4000.0)` matches the Control response at 80 dB SPL to the pre-exposure chinchilla 4 kHz Wave-I amplitude ($1.3684$ $\mu$V), giving $\approx 0.0149$ $\mu$V/AU.

| Condition | 30 dB SPL | 40 dB SPL | 50 dB SPL | 60 dB SPL | 70 dB SPL | 80 dB SPL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 0.2096 | 1.9149 | 9.0726 | 21.5176 | 68.1508 | **91.8332** |
| **Synaptopathy-50** | 0.1284 | 1.1333 | 4.9877 | 11.2915 | 36.8242 | **49.0532** *(53.4% of Control)* |
| **Synaptopathy-25** | 0.0711 | 0.6168 | 2.6129 | 5.7994 | 19.0367 | **25.4909** *(27.8% of Control)* |
| **Selective-Synaptopathy** | 0.2029 | 1.8334 | 8.2708 | 18.4511 | 52.1874 | **65.4712** *(71.3% of Control)* |
| **OHC-Loss** | 0.0001 | 0.0006 | 0.0065 | 0.0655 | 0.6876 | **6.1919** *(~30 dB threshold shift)* |
| **Mixed-Loss** | 0.0000 | 0.0003 | 0.0034 | 0.0339 | 0.3567 | **3.1971** |

![4 kHz Tone-Burst ABR Wave-I Growth](assets/abr_wave_i_growth_4k.png)

#### 8 kHz Tone-Burst ABR Wave-I (30–80 dB SPL)
`carfac_ephys.fit_response_scale_uv_per_au(..., frequency_hz=8000.0)` matches the Control response at 80 dB SPL to the pre-exposure chinchilla 8 kHz Wave-I amplitude ($1.0985$ $\mu$V), giving $\approx 0.0261$ $\mu$V/AU.

| Condition | 30 dB SPL | 40 dB SPL | 50 dB SPL | 60 dB SPL | 70 dB SPL | 80 dB SPL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 0.0033 | 0.0330 | 0.3328 | 2.4910 | 9.9253 | **42.0962** |
| **Synaptopathy-50** | 0.0019 | 0.0193 | 0.1939 | 1.4197 | 5.3846 | **24.3663** *(57.9% of Control)* |
| **Synaptopathy-25** | 0.0011 | 0.0107 | 0.1047 | 0.7580 | 2.8019 | **13.0749** *(31.1% of Control)* |
| **Selective-Synaptopathy** | 0.0032 | 0.0319 | 0.3207 | 2.3775 | 9.1512 | **34.5782** *(82.1% of Control)* |
| **OHC-Loss** | 0.0000 | 0.0000 | 0.0005 | 0.0050 | 0.0501 | **0.5234** *(~30 dB threshold shift)* |
| **Mixed-Loss** | 0.0000 | 0.0000 | 0.0003 | 0.0025 | 0.0256 | **0.2679** |

![8 kHz Tone-Burst ABR Wave-I Growth](assets/abr_wave_i_growth_8k.png)

#### 4/8 kHz Composite Average ABR Wave-I (30–80 dB SPL)
`carfac_ephys.fit_response_scale_uv_per_au(..., frequency_hz=None)` matches the Control composite response at 80 dB SPL to the animal 4/8 kHz composite baseline ($1.2335$ $\mu$V), giving $\approx 0.0184$ $\mu$V/AU.

| Condition | 30 dB SPL | 40 dB SPL | 50 dB SPL | 60 dB SPL | 70 dB SPL | 80 dB SPL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 0.1065 | 0.9740 | 4.7027 | 12.0043 | 39.0380 | **66.9647** |
| **Synaptopathy-50** | 0.0651 | 0.5763 | 2.5908 | 6.3556 | 21.1044 | **36.7098** *(54.8% of Control)* |
| **Synaptopathy-25** | 0.0361 | 0.3138 | 1.3588 | 3.2787 | 10.9193 | **19.2829** *(28.8% of Control)* |
| **Selective-Synaptopathy** | 0.1030 | 0.9326 | 4.2957 | 10.4143 | 30.6693 | **50.0247** *(74.7% of Control)* |
| **OHC-Loss** | 0.0000 | 0.0003 | 0.0035 | 0.0352 | 0.3688 | **3.3576** |
| **Mixed-Loss** | 0.0000 | 0.0002 | 0.0018 | 0.0182 | 0.1912 | **1.7325** |

![4/8 kHz Composite Average ABR Wave-I Growth](assets/abr_wave_i_growth_avg.png)

#### Tone-Burst Response Waveforms at 80 dB SPL
Side-by-side horizontal response waveforms (4 kHz and 8 kHz at 80 dB SPL) demonstrating the compound neural onset and subsequent rate adaptation:

![Compound ABR Tone-Burst Response Waveforms](assets/tone_burst_waveforms.png)

#### Tone-Burst Empirical Validation Against Chinchilla Data
Simulated threshold shifts ($0.1$ $\mu$V crossing) and suprathreshold post/pre ratios from the `Selective-Synaptopathy` cohort compared against noise-exposed chinchilla tone-burst measurements from Bharadwaj et al. (2022):

| Metric | Simulated (Selective-Synaptopathy) | Animal (Bharadwaj et al. 2022) | Tolerance | Status |
| :--- | :---: | :---: | :---: | :---: |
| Tone-Burst 4000 Hz ABR threshold shift (dB) | $+0.55$ | $+1.61$ | $\pm 3$ dB | PASSED |
| Tone-Burst 4000 Hz Wave-I ratio (80 dB SPL) | $0.713$ | $0.626$ | $\pm 0.10$ | PASSED |
| Tone-Burst 8000 Hz ABR threshold shift (dB) | $+0.43$ | $+2.08$ | $\pm 3$ dB | PASSED |
| Tone-Burst 8000 Hz Wave-I ratio (80 dB SPL) | $0.821$ | $0.831$ | $\pm 0.10$ | PASSED |
| Tone-Burst 4/8 kHz average ABR threshold shift (dB) | $+1.11$ | $+1.84$ | $\pm 3$ dB | PASSED |
| Tone-Burst 4/8 kHz average Wave-I ratio (80 dB SPL) | $0.747$ | $0.717$ | $\pm 0.10$ | PASSED |

---

### 3.3 Envelope Following Response (EFR) Growth (SAM Tones: 40–80 dB SPL)

Carrier $f_c = 2000$ Hz, modulation frequency $f_m = 100$ Hz, $100\%$ modulation depth:

| Condition | 40 dB SPL | 50 dB SPL | 60 dB SPL | 70 dB SPL | 80 dB SPL |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control** | 1.0718 | 1.7686 | 2.5543 | 3.6773 | **5.9665** |
| **Synaptopathy-50** | 0.7172 | 1.0091 | 1.5053 | 2.0680 | **2.9614** *(49.6% of Control)* |
| **Synaptopathy-25** | 0.3906 | 0.6391 | 0.7122 | 1.2044 | **1.9453** *(32.6% of Control)* |
| **Selective-Synaptopathy** | 0.9641 | 1.3883 | 1.8917 | 2.9321 | **4.6401** *(77.8% of Control)* |
| **OHC-Loss** | 0.0038 | 0.0382 | 0.3602 | 1.8413 | **4.9530** |
| **Mixed-Loss** | 0.0023 | 0.0230 | 0.2180 | 1.0620 | **2.6369** |

![EFR Growth Curves](assets/efr_growth.png)

---

## 4. Reproduction Steps

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

Verify all 166 unit and integration tests:
```bash
uv run pytest -v
```

### 3. Run Cohort Simulation

Execute simulations across cohorts using the CLI tool:

```bash
# Run both click and tone-burst simulations (default)
uv run carfac-ephys-simulate --output-dir output/ --stimulus all

# Run broadband click and EFR simulations only
uv run carfac-ephys-simulate --output-dir output/ --stimulus click

# Run 4 kHz and 8 kHz tone-burst simulations only
uv run carfac-ephys-simulate --output-dir output/ --stimulus tone-burst
```

This will:
1. Run click-evoked ABR simulations from 30 to 80 dB SPL (when `--stimulus all` or `click`).
2. Run SAM-tone EFR simulations from 40 to 80 dB SPL (when `--stimulus all` or `click`).
3. Run alternating-polarity 4 kHz and 8 kHz tone-burst simulations from 30 to 80 dB SPL (when `--stimulus all` or `tone-burst`).
4. Validate all biological signatures, including quantitative comparisons against empirical chinchilla ABR data.
5. Print summary tables and calibrated scaling factors to stdout.
6. Save publication-quality figures (`abr_wave_i_growth_click.png`, `abr_wave_i_growth_4k.png`, `abr_wave_i_growth_8k.png`, `abr_wave_i_growth_avg.png`, `tone_burst_waveforms.png`, `efr_growth.png`, `empirical_comparison.png`) and `simulation_report.md` to `output/`.

#### Fast Smoke Test
To verify the pipeline on a reduced 2-level subset:
```bash
uv run carfac-ephys-simulate --quick --output-dir output/
```

#### Python API Usage
Simulations can also be invoked directly from Python:

```python
import carfac_ephys as ce

# 1. Generate tone-burst stimuli with alternating polarity
burst_pos = ce.generate_tone_burst(frequency_hz=4000.0, peak_db_spl=80.0, polarity=1.0)
burst_neg = ce.generate_tone_burst(frequency_hz=4000.0, peak_db_spl=80.0, polarity=-1.0)

# 2. Multi-frequency tone-burst cohort simulation (30 to 80 dB SPL)
tb_results = ce.simulate_tone_burst_cohort(tone_burst_levels_db=[30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
print("4 kHz Control @ 80 dB:", tb_results.results_4k["Control"][-1])
print("8 kHz Control @ 80 dB:", tb_results.results_8k["Control"][-1])
print("Composite Avg Control @ 80 dB:", tb_results.composite_results["Control"][-1])

# 3. Empirical comparison against Bharadwaj et al. (2022)
tb_comp = ce.compare_tone_burst_to_empirical(tb_results)
print(ce.format_tone_burst_comparison_table(tb_comp))

# 4. Generate waveform figures and input-output growth plots
wave_4k = ce.simulate_tone_burst_waveforms(frequency_hz=4000.0, level_db=80.0)
wave_8k = ce.simulate_tone_burst_waveforms(frequency_hz=8000.0, level_db=80.0)
ce.plot_tone_burst_waveforms(wave_4k, wave_8k, "output/tone_burst_waveforms.png")
ce.plot_tone_burst_individual_growth(tb_results, "output/")
```

---

## 5. Package Architecture

```text
carfac-ephys/
├── pyproject.toml              # Standalone dependencies and scripts
├── README.md                   # Documentation and walkthrough
├── assets/                     # Published diagnostic figures
│   ├── abr_wave_i_growth.png
│   ├── abr_wave_i_growth_click.png
│   ├── abr_wave_i_growth_tone_burst.png
│   ├── efr_growth.png
│   └── empirical_comparison.png
│   ├── empirical_comparison.png
│   ├── simulation_report.md
│   └── tone_burst_waveforms.png
├── src/
│   └── carfac_ephys/
│       ├── __init__.py
│       ├── constants.py        # Calibration constants (104 dB SPL = 0 dB FS)
│       ├── stimuli.py          # Calibrated click and SAM tone generation
│       ├── stimuli.py          # Calibrated click, SAM tone, and tone-burst generation
│       ├── carfac_model.py     # Biophysical CARFAC wrapper (OHC & fiber retention)
│       ├── electrophysiology.py # ABR Wave-I and EFR metric extractors
│       ├── empirical.py        # Chinchilla ABR dataset loader (Bharadwaj et al. 2022)
│       ├── data/               # Packaged empirical data files
│       ├── experiment.py       # Cohort definitions, level sweeps, validation
│       └── cli.py              # CLI entry point (carfac-ephys-simulate)
└── tests/
    ├── test_stimuli.py
    ├── test_carfac_model.py
    ├── test_electrophysiology.py
    ├── test_empirical.py
    └── test_experiment.py
```

---

## 6. References

- **Bharadwaj HM, Hustedt-Mai AR, Ginsberg HM, et al.** (2022). *Cross-species experiments reveal widespread cochlear neural damage in normal hearing.* Communications Biology, 5(1), 733. [doi:10.1038/s42003-022-03691-4](https://doi.org/10.1038/s42003-022-03691-4).
- **Mehraei G, Hickox AE, Bharadwaj HM, et al.** (2016). *Auditory Brainstem Response Latency in Noise as a Marker of Cochlear Synaptopathy.* Journal of Neuroscience, 36(13), 3755–3764. [doi:10.1523/JNEUROSCI.4460-15.2016](https://doi.org/10.1523/JNEUROSCI.4460-15.2016).
- **Ginsberg HM, Singh R, Bharadwaj HM, Heinz MG.** (2023). *A multi-channel EEG mini-cap can improve reliability for recording auditory brainstem responses in chinchillas.* Journal of Neuroscience Methods, 398, 109954. [doi:10.1016/j.jneumeth.2023.109954](https://doi.org/10.1016/j.jneumeth.2023.109954).
- **Dolphin WF, Mountain DC.** (1992). *The envelope following response: Scalp potentials elicited in the Mongolian gerbil using sinusoidally AM acoustic signals.* J. Acoust. Soc. Am., 92(1), 86–96. [doi:10.1121/1.404225](https://doi.org/10.1121/1.404225).
- **Lyon RF.** (2017). *Human and Machine Hearing: Extracting Meaning from Sound.* Cambridge University Press.
