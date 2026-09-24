# Changelog

## 0.2.0 — unreleased

### Changed — public API

The loader now serves more than one study design, so the chinchilla-specific names have
been replaced:

| Removed | Replacement |
| :--- | :--- |
| `load_chinchilla_abr_dataset()` | `load_abr_dataset(species="chinchilla")` |
| `ChinchillaAbrDataset` | `AbrDataset` |
| `PrePostStat` | `GroupComparisonStat` |
| `AnimalWaveAmplitudes` | `SubjectWaveAmplitudes` |
| `.mean_pre` / `.mean_post` | `.mean_baseline` / `.mean_comparison` |
| `.shift` / `.ratio` | `.difference` / `.ratio_of_means` |
| `.per_animal_w1_ratios` | `.paired_w1_ratios` |
| `.high_level_w1_uv[0.0].mean_pre` | `.baseline_reference_w1_uv` |

### Added

- `AbrDataset.design`, `PAIRED_TIMEPOINTS` and `INDEPENDENT_GROUPS`.
- `context_measures` for non-ABR measures, each carrying its own units.