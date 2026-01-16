# Validation Datasets

This directory contains synthetic datasets used for multi-configuration robustness validation (see paper.html Section 5.6).

## Datasets

### Small Sample (n=12 sprints)
- **small_low_cv.csv**: 7-day sprints, stable team (CV=0.088)
  - Scenario: New team, stable environment
  - Use case: Minimum viable dataset testing
  
- **small_high_cv.csv**: 7-day sprints, high disruption (CV=0.184)
  - Scenario: New team, reactive mode
  - Use case: High-volatility impact testing

### Medium Sample (n=24 sprints)
- **medium_mixed.csv**: Mixed sprint lengths (7/14 days), moderate volatility (CV=0.093)
  - Scenario: Mature team, variable cadence
  - Use case: Production-ready threshold testing

### Large Sample (n=48 sprints)
- **large_stable.csv**: 14-day sprints, moderate volatility (CV=0.183)
  - Scenario: Established team, 2-week cadence
  - Use case: Long history, standard Scrum

- **large_volatile.csv**: Mixed sprint lengths, moderate disruption (CV=0.114)
  - Scenario: Long history, occasional chaos
  - Use case: Realistic team dynamics

### Extra Large Sample (n=96 sprints)
- **xlarge_realistic.csv**: Realistic patterns (CV=0.194)
  - Scenario: Multi-year history
  - Use case: Convergence testing, diminishing returns

## Generation

All datasets generated via:
```bash
python generate_dataset.py --n <N> --seed <SEED> --unit person_days [--fixed-sprint-weeks <W>]
```

Parameters calibrated to produce:
- Throughput CV: 0.09-0.19 (literature range: 0.08-0.35)
- Bug fix %: 8.5±3% (literature range: 10-15%)
- Unplanned work: 0.9-2.5% (conservative vs literature 5-20%)

See paper.html Section 5.7 for generator justification and literature validation.

## Running Validation

```bash
python run_validation_experiments.py
```

Produces:
- `validation_results.csv`: Summary table
- `*_history/`: Forecast artifacts per configuration
- `*_model.json`: Trained model parameters
- `*_config.json`: Forecast configuration

## Key Findings

See `VALIDATION_SUMMARY.md` for executive summary.

**TL;DR:**
- 70% forecast improvement: n=12 → n=24 sprints
- 49% wider forecasts when CV doubles
- Stable beyond n≥24 (diminishing returns)
- Sprint length irrelevant (7-day vs 14-day)
