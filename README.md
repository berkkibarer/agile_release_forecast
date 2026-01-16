# Agile Release Forecasting & Sprint Analytics Pipeline — README

## Purpose and Problem

This system provides, **without using story points**, using only **effort-based (person-days)** data:
- Release forecasts (median, p90, finish date)
- Sprint metric analysis (throughput, volatility, plannedness, carryover, burnout)
- Risk indicators and operational health metrics
- Calendar-aware (weekend, holiday-aware) Monte Carlo simulations
- Automated visualization (burndown, burnup, throughput trends, burnout)
- HTML report generation (for Product Owner and team)

**Goal:** Extract all the benefits that Scrum provides with story points (uncertainty, risk, predictability) in an effort-based manner.

---

## Content / Files

### Main Pipeline Scripts:
- `forecast_release.py` — Main forecasting engine (OLS regression + Sequential Monte Carlo simulation)
  - **Uncertainty management:** Two-layer uncertainty propagation
    1. **Parameter uncertainty**: β ~ MVN(β_hat, Σ) - model parameter uncertainty (sampling from covariance matrix)
    2. **Residual uncertainty**: ε ~ N(0, σ²) - sprint-to-sprint stochastic variation
  - **Low-data fallback**: deterministic mode for n < 3 sprints (observed rate + heuristic sigma)
- `compute_effort_metrics.py` — Utility for computing sprint metrics (throughput, volatility, plannedness, carryover, burnout)
- `agile_plots.py` — Visualizations (burndown, burnup, throughput trend, burnout chart)
- `generate_report.py` — HTML report generator (single file, all metrics + charts)
- `generate_dataset.py` — Synthetic sprint data generator (for testing/demo/validation)
- `summarize_sims.py` — Simulation results summary (percentile analysis)

### Validation & Research:
- `run_validation_experiments.py` — Multi-configuration validation automation (see paper Section 5.6)
- `compute_remaining_effort.py` — Helper utility for dataset parameterization
- `validation_datasets/` — Synthetic datasets for robustness testing
  - `VALIDATION_SUMMARY.md` — Executive summary of validation findings
  - `*.csv` — Six configurations (small/medium/large × low/high volatility)
- `paper.html` — Academic paper with validation study, generator justification, and deployment guidance

### Configuration:
- `forecast_config.json` — Main config file (example)

---

## Computed Metrics (All Scrum Needs)

### 1. Throughput & Velocity
- **throughput_pd_daily**: Daily net work output (person-days/day)
- **net_done**: Net effort completed in sprint (velocity - scope_added)
- **effective_days**: Working days (excluding weekends and holidays)

### 2. Volatility & Predictability
- **throughput_cv_w**: Rolling CV (std/mean) over window (default 6 sprints)
  - High CV → low predictability, high uncertainty
- **throughput_std_w**: Rolling standard deviation

### 3. Plannedness (Plan Disruption)
- **unplanned_fraction**: scope_added / velocity
  - High value → sprint plan breaking down, reactive work
- Target: <15% (KPI threshold)

### 4. Carryover (Estimation Error)
- **carryover_pd**: Work taken into sprint but not completed (committed_pd - net_done)
- **carryover_ratio**: carryover_pd / committed_pd
  - High value → estimation error or blockers
- Target: <20%

### 5. Rework & Quality
- **percent_bug**: Bug fix rate (0-1 range)
- **rework_fraction**: Rework rate (if tracking available)
- Target: <10%

### 6. Workload & Burnout
- **workload_ratio**: velocity / (team_size × effective_days)
  - >1.0 → team working above capacity
- **burnout_index**: Rolling mean of workload_ratio
  - >0.85 → burnout risk
- Target: 0.70-0.85 for sustainability

### 7. Forecast Outputs
- **Median estimate**: 50% probability finish (sprint count and date)
- **P90 (Conservative)**: 90% probability finish (for conservative planning)
- **Percentile distribution**: P10, P25, P50, P75, P90, P95
- **Remaining effort**: Remaining work (person-days)

---

## Method and Technical Details

### 1. Effort Metrics Computation (`compute_effort_metrics.py`)
**Inputs:**
- Sprint CSV (sprint_id, start_date, end_date, velocity, scope_added, team_size, percent_bug, committed_pd [optional])

**Processing Steps:**
1. **Calendar days calculation**: end_date - start_date + 1
2. **Weekend days counting**: Saturday/Sunday automatically detected
3. **Effective days**: calendar_days - weekend_days - holidays
4. **Net done derivation**: max(0, velocity - scope_added)
5. **Throughput**: net_done / effective_days
6. **Rolling stats**: mean, std, CV with 6-sprint window (configurable)
7. **Carryover derivation**: If committed_pd exists → max(0, committed_pd - net_done)
8. **Workload**: velocity / (team_size × effective_days) if team_size exists

**Outputs:**
- Enriched DataFrame (20+ derived columns)
- `history/sprint_features.csv` (saved as artifact)

### 2. Forecasting Engine (`forecast_release.py`)

**Method: Sequential Monte Carlo + OLS Regression**

#### A. Regression Model (OLS)
**Target variable (y):** `daily_rate` (net_done / effective_days)

**Explanatory variables (X):**
- `prev_daily_rate`: Previous sprint throughput (lag feature)
- `unplanned_fraction`: Plan disruption rate
- `percent_bug`: Bug rate
- `throughput_cv_w`: Volatility (rolling CV)

**Model:**
```
daily_rate = β₀ + β₁·prev_daily_rate + β₂·unplanned_fraction + β₃·percent_bug + ε
```

**Parameter uncertainty:**
- Beta parameters: Sampled from MVN(β_hat, Σ) ~ Multivariate Normal distribution
- Covariance matrix (Σ): Computed from OLS fit
- Residual uncertainty (ε): N(0, σ²)

#### B. Monte Carlo Simulation (Sequential)
**Each simulation (5000 times):**

1. **Parameter sampling**: β* ~ MVN(β_hat, Σ)
2. **Sprint-by-sprint progression:**
   ```
   remaining_work = total_release_effort - cumulative_done
   
   For each sprint:
     - Calendar pattern: Use pattern from last 6 sprints
     - Weekend/holiday: Auto-calculate
     - Feature vector: Random sample from historical sprints
     - Predicted rate: x·β* + ε (ε ~ N(0, σ))
     - Sprint capacity: predicted_rate × effective_days
     - remaining_work -= capacity
     - If remaining_work ≤ 0: calculate finish date, stop
   ```
3. **Output**: days_needed, sprints_needed, finish_date

#### C. Low-Data Fallback (n < 3)
If historical data <3 sprints:
- OLS is skipped
- Last sprint's daily_rate → used as intercept
- Parameters: β = [observed_rate, 0, 0, ...]
- Sigma: Heuristic (0.5 × observed_rate)
- Flag: `low_data_mode: true` written to artifact

**Why:** 1-2 rows of data are insufficient for OLS, fallback is deterministic and safe.

### 3. Visualization (`agile_plots.py`)
**4 basic charts:**

1. **Burndown**: Remaining effort vs time
   - Y-axis: Remaining effort (person-days)
   - X-axis: Sprint end dates
   
2. **Burnup**: Cumulative done vs target
   - Red line: Release target
   - Blue line: Cumulative progress
   
3. **Throughput Trend**: Daily throughput + volatility
   - Main line: throughput_pd_daily
   - Dashed line: Rolling mean
   - Shaded area: ±1σ band
   
4. **Burnout**: Workload ratio + burnout index
   - Workload ratio: Capacity utilization rate
   - Burnout index: Rolling average
   - Red line: 100% capacity threshold

### 4. Report Generation (`generate_report.py`)
**In a single HTML file:**
- Executive summary (remaining, median, p90 estimates)
- Sprint data snapshot (first 6 rows)
- Numeric column statistics (mean, std, percentiles)
- Model parameters (β coefficients)
- Simulation results (percentile distribution)
- All charts (embedded base64 PNG)
- Automated commentary:
  - High bug rate warning
  - Volatility alert
  - Carryover issues
  - Unplanned work commentary
- Links to raw artifacts (CSV, JSON)

---

## 📁 CSV Schema

### Required Columns:

---

## CSV schema — Minimal (fixed)
Recommended and required columns:

Required (must be in every CSV)
- `sprint_id` — string
- `start_date` — YYYY-MM-DD
- `end_date` — YYYY-MM-DD
- `velocity` — top-level unit (points OR person_days)
- `scope_added` — top-level unit (same unit)
  
Recommended (for model/features)
- `team_size` — integer (optional but useful)
- `percent_bug` — float 0..1 (optional but useful)

Derived (scripts auto-calculate — can be in CSV but not required)
- `calendar_days`, `weekend_days`, `effective_days`, `net_done`, `daily_rate`, `prev_daily_rate`

Recommended to DROP/confusing columns (default pipeline minimal)
- `holidays_in_sprint` (holidays will be provided via config)
- `team_daily_prod_points`, `points_per_person_per_day`
- `velocity_person_days`, `scope_added_person_days`, `net_done_person_days`
- `blocker_hours`, `code_churn`, `avg_story_size` (optional)

---

## Config — Minimal example

The following `forecast_config_minimal.json` is the recommended minimal structure:

```json
{
  "csv_path": "example_sprints_1w_pd.csv",
  "total_release_effort": 450,
  "effort_unit": "person_days",
  "save_model_path": "model_artifact.json",
  "history_path": "history/model_history.jsonl",
  "use_features": ["scope_added", "percent_bug", "prev_daily_rate"],
  "future_holiday_dates": [],
  "output_paths": {
    "sims_csv": "sims_output_regression.csv",
    "model_artifact": "model_artifact.json",
    "selection_csv": "model_selection_results.csv"
  }
}
```

Explanations:
- `effort_unit` must match the top-level unit in the CSV (`points` or `person_days`).
- `future_holiday_dates` is an array of future holiday dates in YYYY-MM-DD format.
- `use_features`: feature names to use in the model. If `prev_velocity` is an old name, the script automatically maps it to `prev_daily_rate`.

---

Throughput (points per effective day): velocity_points / effective_days
shows: how many points the team completes daily.
Points volatility (std, CV = std / mean) over window (e.g. last 6 sprints)
high CV → low predictability.
Predictability / Plannedness: planned_points / velocity_points or unplanned_fraction = scope_added_points / velocity_points
high unplanned_fraction → plan breaking down.
Avg story size & story_size_std (if story-level data available)
large or high std → high risk.
Carryover ratio: points carried to next sprint / sprint commitment
high carryover → estimation error / blockers.
Rework / bug ratio: percent_bug or rework_points / velocity_points
quality impact, productivity loss.
Lead time / cycle time distribution (if ticket-level time available)
long tail → uncertainty.
Blocked time fraction (if tracked)
operational risk indicator.
Points-per-person-day (historical ppd) distribution — uncertainty measure (calculate with person_days reference if available)

## Basic usage

1) Data generator (example):
- Points top-level:
  ```
  python3 generate_dataset.py --out example_sprints_minimal_points.csv --n 120 --seed 42 --unit points
  ```
- Person-days top-level, 1-week sprints:
  ```
  python3 generate_dataset.py --out example_sprints_minimal_1w_pd.csv --n 120 --seed 42 --unit person_days --fixed-sprint-weeks 1
  ```

2) Forecast (with default MC parameters):
```
python3 forecast_release.py --config forecast_config.json
```
- CLI override:
  - `--n-sims 2000`
  - `--recent-sprint-window 4`

---

## Low-data behavior (automatic)
- If history (CSV row count) < 3:
  - OLS fit is skipped.
  - `observed_rate = last_row.daily_rate` will be the intercept value used.
  - `beta` = [observed_rate, 0, 0, ...] ; cov = None.
  - `sigma` is set with conservative heuristic (e.g. `sigma = max(0.5 * observed_rate, 0.5)`).
  - Script saves `low_data_mode` flag in `model_artifact.json` and prints warning to console.

Why: single/two-row history doesn't provide reliable OLS parameter/covariance. Fallback provides deterministic and safe behavior.

---

## Models and uncertainty (brief)
- Default: `daily_rate` (net_done / effective_days) regressed with OLS (statsmodels); parameter covariance matrix computed.
- Monte-Carlo:
  - MVN (beta ~ N(beta_hat, cov)) sampled for parameters (if cov available).
  - Residual (sigma) added in each sprint sample.
- Alternative/additional options:
  - Non-parametric bootstrap (row-resample) — `run_bootstrap.py`.
  - Ridge / regularization (to be added optionally).
  - Full Bayesian (MCMC) only if needed — high cost.

Recommendation: for small data (e.g. n <= 12) use p90 for conservative planning.

---

## Tests / Sensitivity studies
- Recent-window sensitivity:
  ```
  python3 forecast_release.py --config forecast_config.json --recent-sprint-window 4 --n-sims 2000
  python3 forecast_release.py --config forecast_config.json --recent-sprint-window 6 --n-sims 2000
  ```
  Compare: median/p90 days and sprint counts.

- n_sims stability:
  ```
  python3 forecast_release.py --config forecast_config.json --n-sims 1000
  python3 forecast_release.py --config forecast_config.json --n-sims 10000
  ```
---

## Validation & Testing

### Multi-Configuration Validation
The project includes automated validation across diverse synthetic datasets to verify model robustness:

**Run validation experiments:**
```bash
python3 run_validation_experiments.py
```

**What it does:**
- Tests model on 6 different configurations (n=12-96 sprints, varying volatility)
- Generates forecasts for each configuration (5000 MC simulations each)
- Produces summary table with forecast precision metrics
- Validates sample size effects, volatility impacts, and sprint length independence

**Key findings (see `validation_datasets/VALIDATION_SUMMARY.md`):**
- Forecast precision improves **70%** when moving from n=12 to n=24 sprints
- Volatility (CV) is primary driver of forecast width (**49% increase** when CV doubles)
- Model stable beyond n≥24 sprints (diminishing returns)
- Sprint length (7-day vs 14-day) does not affect forecast quality

**Validation datasets:**
- `small_low_cv`: 12 sprints, low volatility (CV=0.088)
- `small_high_cv`: 12 sprints, high volatility (CV=0.184)
- `medium_mixed`: 24 sprints, mixed length
- `large_stable`: 48 sprints, 14-day cadence
- `large_volatile`: 48 sprints, mixed, volatile
- `xlarge_realistic`: 96 sprints, realistic pattern

### Synthetic Data Generator Validation
The generator (`generate_dataset.py`) produces theory-driven synthetic data incorporating:
- **AR(1) momentum** (ρ=0.45) for sprint-to-sprint correlation
- **Seasonal variation** (26-sprint period) modeling organizational rhythms
- **Team scaling effects** (Brooks' Law adjustment)
- **Quality tax** (bug percentage impact)
- **Stochastic noise** (σ=1.8 pd/day for unpredictable events)

Generated data validated against empirical Agile literature (Vacanti 2015, Sutherland 2014, McConnell 2006). See **paper.html Section 5.7** for detailed justification.

---

## Errors / Troubleshooting
- Error: `CSV missing required columns` — check CSV headers (sprint_id,start_date,end_date,velocity,scope_added)
- Error: finish_date falls on weekend — check `future_holiday_dates` or sprint start logic; `add_workdays` & weekend calculation is done automatically in script.
- If you see `kurtosistest` warning: statistical tests unreliable due to small sample (n<20) — normal.

---

## Operational recommendations
- Retrain the model after each sprint.
- Enter `total_release_effort` in correct unit (must match config.effort_unit).
- Use minimal CSV schema; let scripts calculate derived columns.
- Prefer conservative p90 in small data situations.
- **For production use:** Target n≥20-24 historical sprints for stable forecasts.
- **High-volatility teams (CV>0.15):** Target n≥30 for comparable precision.