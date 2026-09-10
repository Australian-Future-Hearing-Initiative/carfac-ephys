Reproduce animal model cochlear impairment electrophysiology (ABR Wave-I and EFR level series) in silico using the CARFAC model in the isolated package carfac-ephys.

Success Criteria:
1. Isolated Package: All code and tests reside strictly in carfac-ephys/ with zero modifications to hp-acoustic. Managed via uv.
2. Biophysical Impairment Modeling: Clean CARFAC interface supporting outer hair cell degradation (ohc_health in [0, 1]) and auditory nerve deafferentation / synaptopathy (fiber_retention in [0, 1] across HSR/MSR/LSR).
3. Calibrated Stimulus Suite: Click generator (100 us rectangular pulse, 30-80 dB SPL) and SAM tone generator (fc=2000 Hz, fm=100 Hz, 40-80 dB SPL) calibrated against CARFAC digital reference (104 dB SPL = 0 dB FS).
4. Electrophysiology Extraction: Algorithms to compute population auditory nerve response, onset peak-to-trough amplitude for ABR Wave-I / CAP, and spectral Fourier magnitude at fm for EFR.
5. Biological Signature Reproduction:
   - Synaptopathy preserves low-level thresholds (30-40 dB SPL) while causing proportional reduction in suprathreshold Wave-I and EFR amplitudes.
   - OHC loss elevates thresholds (no response at 30-50 dB SPL) and shifts I/O curves rightward with loss of compression.
6. Test Verification: 100% test pass rate via uv run pytest tests/.
7. Demonstration CLI: carfac-ephys-simulate CLI running the cohort comparison, printing summary tables, and outputting diagnostic figures.
